import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import math

from torch.autograd import Variable

# 优化器
from pyitcast.transformer_utils import get_std_opt
# 标签平滑
from pyitcast.transformer_utils import LabelSmoothing
# 损失计算
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import Batch


def subsequent_mask(size):
    attn_shape = (1, size, size)
    # 先生成上三角矩阵
    # np.triu 将第k条对角线下的元素置为0
    upper = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 三角阵反转 得到需要的下三角矩阵
    return torch.from_numpy(1 - upper)


def attention(query, key, value, mask=None, dropout=None) -> (torch.Tensor, torch.Tensor):
    """ mask: 掩码张量, dropout: nn.Dropout对象 """
    # 取 query 的最后一维 即词嵌入维度
    d_k = query.size(-1)
    # 计算注意力矩阵 K的转置为最后两个维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否需要掩码张量 Decoder中会使用到 mask
    if mask is not None:
        # 使用 masked_fill API, 进行比较并替换为 -INF 这样softmax后为0
        scores = scores.masked_fill(mask == 0, 1e-9)

    # 对 scores 最后一维进行 softmax 操作
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否需要 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后和 value 相乘 并返回注意力矩阵
    return torch.matmul(p_attn, value), p_attn


# 因为多头注意力机制中存在多个结构相同的线性层 因此需要克隆
def clones(module, N: int) -> nn.ModuleList:
    """用于生成相同网络层的克隆函数 module代表要克隆的目标网络层 N代表克隆的数目"""
    # 对 module 进行 N 次拷贝
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        """ vocab: 词表大小 """
        super(Embedding, self).__init__()
        # 直接调用API
        self.ebd = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.ebd(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """ d_model: 词嵌入维度  dropout: 置0比例  max_len: 句子最大长度 """
        super(PositionalEncoding, self).__init__()

        # dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 位置编码矩阵初始化
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵 绝对位置即词汇的索引 position: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # 将绝对位置矩阵变换为[max_len, d_model]，然后赋值给 pe 就完成了位置编码
        # 因此需要一个变换矩阵来完成该操作 div_term
        # 为了保证梯度收敛速度，需要将绝对位置索引变换为足够小的数字
        # 采用的公式: PE(2i) = sin(pos/10000^(2i/d_model)), PE(2i+1) = cos(pos/10000^(2i/d_model))
        # 将 pos / 10000^(2i/d_model) 变换为 exp 表示即为 pos * e^(-2i*ln(10000)/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 得到 pe: [max_len, d_model], 为了保证和输入同shape 扩展0维度 [1, max_len, d_model]
        pe.unsqueeze(0)

        # pe 不会随着训练/优化而更新，所以注册为buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x为词嵌入层的输出 """
        # 为了和 x 长度适配，对 pe 进行切片 pe[:, :x.size(1)]
        # 同时 pe 不参与梯度求解，因此通过 Variable 进行封装
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        # 断言确保 embedding 整除 head
        assert embedding_dim % head == 0

        # 每个头获得的词向量维度
        self.d_k = embedding_dim // head

        # 传入 head
        self.head = head

        # 4个权重矩阵 Wq Wk Wv W0 shape均为[ebd, ebd]
        # W0为最后concat后再经过的线性层
        self.liners = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 最后得到的注意力张量
        self.attn = None

        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # LayerNorm
        self.ln = nn.LayerNorm(embedding_dim, eps=1e-5)

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数 Q K V原始输入以及 mask矩阵"""

        # mask处理
        if mask is not None:
            # 使用 unsqueeze 扩展维度 即header这一维度 代表多头中第n个头
            mask = mask.unsqueeze(1)

        # 获取 batch_size
        batch_size = query.size(0)

        # 利用 zip 将 QKV 与三个线性层组合 并通过将 QKV 传入线性层得到对应线性层输出
        # 同时需要保证shape为[batch_size, head, max_len, embedding_dim] 需要view()+transpose()
        # query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        #                              for model, x in zip(self.liners, (query, key, value))]
        # 更容易理解的写法
        query = self.liners[0](query).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        key = self.liners[1](key).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        value = self.liners[2](value).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        # 调用之前实现的attention函数 因为函数中只利用了tensor的最后两维进行计算
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到了每个头计算的4d的结果 因此为了保证经过线性层以及后续维度不变，需要concat 即transpose()+view()
        # view()必须保证张量内存连续 因此需要contiguous()方法
        # x shape: [batch_size, max_len, embedding_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # concat 后的 tensor 再经过一个线性层 W0
        # 然后残差
        x = x + self.dropout(self.liners[-1](x))

        # layerNorm
        return self.ln(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """d_model: 词嵌入维度也是第一个层输入维度 d_ff: 第一个线性层输出维度"""
        super(FeedForward, self).__init__()

        # 实例化两个线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 实例化dropout
        self.dropout = nn.Dropout(p=dropout)
        # 实例化LayerNorm
        self.ln = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        """ x为 Attention 层输出 """
        # 首先和 W1 进行矩阵乘，然后通过 ReLU 激活函数 ReLU(x)=max(0, x)
        # 然后进行dropout，和 W2 进行矩阵乘， 得到结果
        return self.ln(x + self.w2(self.dropout(F.relu(self.w1(x)))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """ features: 词嵌入维度
            eps: 规范化公式分母，防止分母为0 """
        super(LayerNorm, self).__init__()

        # 根据features形状初始化参数张量 a2,b2 一个全1 一个全0
        # 作用：若直接对上一层输出进行规范化，将改变结果的正常表征，因此需要参数作为调节因子，使得其既能满足规范化要求，又不改变针对目标的表征
        # 通过 nn.Parameter 封装，代表模型参数，即需要训练
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 传入 eps
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """ x为上一层输出 """
        # 首先对输入变量求最后一个维度的均值，然后求最后一个维度的标准差，并保持输出维度不变
        # 然后根据规范化公式进行计算 x - mean / std
        # 对结果点乘缩放参数a2 加上位移参数b2 */+为同型点乘/加
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    """ 子层连接结构/残差连接 """

    def __init__(self, size, dropout=0.1):
        """size一般为词嵌入的维度"""
        super(SublayerConnection, self).__init__()

        # 实例化规范化层对象
        self.norm = LayerNorm(size)
        # 实例化dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        sublayer: 该子层连接中的子层函数
        e.g. sublayer可以代表 attention层
        """
        # 对输出进行初始化 然后交给子层处理 再对子层进行dropout
        # 相当于残差连接 未处理的x和处理后的x进行相加
        # return x + self.dropout(sublayer(self.norm(x)))) 有错
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头注意力层的实例化对象
        :param feed_forward: MLP层的实例化对象
        :param dropout: 丢弃比例
        """
        super(EncoderLayer, self).__init__()

        # 传入对应实例化对象
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 编码器有两个子层连接结构 因此通过clones函数进行克隆
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)

        # 传入size
        self.size = size

    def forward(self, x, mask):
        """
        :param x: 前一层输出
        :param mask: 掩码张量
        :return: 编码器输出
        """
        # 首先是多头注意力层 然后是mlp层
        # x = self.sublayer[0](x, lambda qkv: self.self_attn(qkv, qkv, qkv, mask))
        # return self.sublayer[1](x, self.feed_forward)
        x = self.self_attn(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 编码器层
        :param N: 层数
        """
        super(Encoder, self).__init__()
        # 克隆N层编码器层
        self.layers = clones(layer, N)
        # 初始化规范化层 放在编码器最后
        # 这里的layer.size为词嵌入维度
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头自注意力对象 Q=K=V
        :param src_attn: 多头注意力对象 Q!=K=V
        """
        super(DecoderLayer, self).__init__()
        # 传入参数
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 三个子层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输出
        :param memory: 来自编码器层的语义存储变量
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        """
        # 多头自注意力层
        # 对目标数据进行遮掩 保证解码器能看到的信息不包括未来的信息
        x = self.sublayer[0](x, lambda qkv: self.self_attn(qkv, qkv, qkv, target_mask))

        # 多头注意力层
        # query为解码器输入 k v为编码器的输出
        # 这里的遮掩是对源数据进行遮掩 并不是为了抑制解码器的视野 只是为了遮蔽无意义的字符如P 编码器中的mask也是这个用途
        x = self.sublayer[1](x, lambda query: self.src_attn(query, memory, memory, source_mask))

        # mlp层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 解码器层
        :param N: 层数
        """
        super(Decoder, self).__init__()
        # 实例化
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 前一层的输出
        :param memory: 编码器输出
        :param source_mask: 源数据掩码
        :param target_mask: 目标数据掩码
        """
        # 对每一层进行循环即可
        for layer in self.layers:
            layer(x, memory, source_mask, target_mask)
        return x


# 线性层+softmax层 生成器类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词嵌入维度
        :param vocab_size: 词表长度
        """
        super(Generator, self).__init__()
        # 线性层
        self.liner = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 首先对x进行线性变换 然后进行softmax
        return F.softmax(self.liner(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_ebd, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_ebd: 源数据嵌入函数
        :param target_ebd: 目标数据嵌入函数
        :param generator: 生成器对象
        """
        super(EncoderDecoder, self).__init__()
        # 传参
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_ebd = target_ebd
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 源数据掩码
        :param target_mask: 目标数据掩码
        """
        # 先获得编码器输出 然后协同解码器输入和掩码张量一同输入解码器
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_ebd(target), memory, source_mask, target_mask)


def make_model(source_vocab=11, target_vocab=11, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    :param source_vocab: 源数据特征总数
    :param target_vocab: 目标数据特征总数
    :param N: 解码器和编码器堆叠次数
    :param d_model: 词嵌入维度
    :param d_ff: mlp矩阵变换维度
    :param head: 多头注意力层的多头数
    :param dropout: 置零比例
    """
    # 准备一个深拷贝
    c = copy.deepcopy

    # 实例化多头注意力对象
    attn = MultiHeadedAttention(head, d_model, dropout)

    # 实例化mlp对象
    mlp = FeedForward(d_model, d_ff, dropout)

    # 实例化位置编码对象
    pe = PositionalEncoding(d_model, dropout)

    # 堆叠模型
    # nn.Sequential 将两个层合为一个
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(mlp), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(mlp), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(pe)),
        nn.Sequential(Embedding(d_model, target_vocab), c(pe)),
        Generator(d_model, target_vocab)
    )

    # 初始化模型参数
    # 如果参数维度大于1 就会初始化为一个服从均匀分布的矩阵 如线性层中的变换矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def data_generator(V, batch, num_batch):
    """
    :param V: 随机生成数字的最大值+1
    :param batch: 每次输入的数据量
    :param num_batch: 输入的次数
    """
    for i in range(num_batch):
        # 随机整数[1, V)
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        # 矩阵第一列设置为1 作为起始标志列 即start
        data[:, 0] = 1

        # copy任务 source和target一致 且样本变量不需要求梯度
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        # 使用Batch对src和tgt进行对应的掩码张量生成 用yield返回
        yield Batch(source, target)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        # 训练模式
        model.train()
        # batch 20
        run_epoch(data_generator(10, 8, 20), model, loss)
        # 评估模式
        model.eval()
        # batch 5
        run_epoch(data_generator(10, 8, 5), model, loss)


if __name__ == '__main__':
    md = make_model()
    model_opt = get_std_opt(md)
    criterion = LabelSmoothing(size=10, padding_idx=0, smoothing=0.0)
    loss = SimpleLossCompute(md.generator, criterion, model_opt)

    run(md, loss)

    # emd = 512
    # head = 8
    # d_model = 512
    # d_ff = 2048
    # attn = MultiHeadedAttention(head, emd)
    # mlp = FeedForward(d_model, d_ff)
    # dropout = 0.2
    # layer = EncoderLayer(emd, attn, mlp, dropout)
    # N = 6
    # # mask = Variable(torch.zeros(8, 4, 4))
    #
    # en = Encoder(layer, N)
    # x = Variable(torch.randn(1, 16, 512))
    # en_result = en(x, None)
    # print(en_result)
    # print(en_result.shape)
