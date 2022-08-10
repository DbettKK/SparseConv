import numpy as np

d_model = 512
d_ff = 2048
max_sen_len = 64
source_vocab = 120
target_vocab = 120
dtype = 'float16'
N = 6


def transformer():
    data_path = '../../data/transformer'
    # Wq,k,v: [d_model, d_model] W0: [d_model, d_model]
    # Wff: [d_model, d_ff], W_model: [d_ff, d_model]
    # Webd: [max_sen_len, d_model]
    # W_last: [d_model, max_sen_len]
    Webd = np.random.uniform(0, 0.1, (source_vocab, d_model)).astype(dtype)
    W_last = np.random.uniform(0, 0.1, (d_model, target_vocab)).astype(dtype)
    Webd.tofile(data_path + '/w_ebd')
    W_last.tofile(data_path + '/w_last')
    for i in range(N):
        en_or_de = 'en'
        Wq = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wk = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wv = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        W0 = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wff = np.random.uniform(0, 0.1, (d_model, d_ff)).astype(dtype)
        W_model = np.random.uniform(0, 0.1, (d_ff, d_model)).astype(dtype)
        Wq.tofile(data_path + '/wq_' + en_or_de + str(i))
        Wk.tofile(data_path + '/wk_' + en_or_de + str(i))
        Wv.tofile(data_path + '/wv_' + en_or_de + str(i))
        W0.tofile(data_path + '/w0_' + en_or_de + str(i))
        Wff.tofile(data_path + '/wff_' + en_or_de + str(i))
        W_model.tofile(data_path + '/wm_' + en_or_de + str(i))
    for i in range(N):
        en_or_de = 'de'
        Wq_self = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wq_src = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wk_self = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wk_src = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wv_self = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wv_src = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        W0_self = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        W0_src = np.random.uniform(0, 0.1, (d_model, d_model)).astype(dtype)
        Wff = np.random.uniform(0, 0.1, (d_model, d_ff)).astype(dtype)
        W_model = np.random.uniform(0, 0.1, (d_ff, d_model)).astype(dtype)
        Wq_self.tofile(data_path + '/wq_self_' + en_or_de + str(i))
        Wq_src.tofile(data_path + '/wq_src_' + en_or_de + str(i))
        Wk_self.tofile(data_path + '/wk_self_' + en_or_de + str(i))
        Wk_src.tofile(data_path + '/wk_src_' + en_or_de + str(i))
        Wv_self.tofile(data_path + '/wv_self_' + en_or_de + str(i))
        Wv_src.tofile(data_path + '/wv_src_' + en_or_de + str(i))
        W0_self.tofile(data_path + '/w0_self_' + en_or_de + str(i))
        W0_src.tofile(data_path + '/w0_src_' + en_or_de + str(i))
        Wff.tofile(data_path + '/wff_' + en_or_de + str(i))
        W_model.tofile(data_path + '/wm_' + en_or_de + str(i))


if __name__ == '__main__':
    transformer()
