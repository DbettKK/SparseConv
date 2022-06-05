all: spmma

spmma: test/test.cu
	nvcc ./test/test.cu ./entity/MatrixParam.cu ./entity/ConvParam.cu ./entity/Tensor4d.cu ./spmma/sparse_matmul.cu ./spmma/sparse_conv.cu ./utils/util.cu ./utils/CudaTime.cu -lcusparse -lcusparseLt

main: main.cu
	nvcc -c ./interface/interface.cu ./spmma/entity/MatrixParam.cu ./spmma/entity/ConvParam.cu ./spmma/entity/Tensor4d.cu ./spmma/kernels/sparse_matmul.cu ./spmma/kernels/sparse_conv.cu ./spmma/utils/util.cu ./spmma/utils/CudaTime.cu -lcusparse -lcusparseLt -Xcompiler '-fPIC'

generate:
	g++ -shared -o main.so *.o -L/usr/local/cuda/lib64 -lcusparse -lcusparseLt

# -fPIC 与位置无关进行编译

clean:
	rm -rf *.o *.so


.PHONY: generate clean