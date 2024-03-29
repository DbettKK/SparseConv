all: spmma

spmma: spmma/test/test.cu
	nvcc ./test/test.cu ./entity/MatrixParam.cu ./entity/ConvParam.cu ./entity/Tensor4d.cu ./spmma/sparse_matmul.cu ./spmma/sparse_conv.cu ./utils/util.cu ./utils/CudaTime.cu -lcusparse -lcusparseLt
	nvcc -c ./interface/interface.cu ./spmma/entity/MatrixParam.cu ./spmma/entity/ConvParam.cu ./spmma/entity/Tensor4d.cu ./spmma/kernels/sparse_matmul.cu ./spmma/kernels/sparse_conv.cu ./spmma/utils/util.cu ./spmma/utils/CudaTime.cu -lcusparse -lcusparseLt -Xcompiler '-fPIC'

tmp:spmma/tmp.cu
	nvcc -c -arch=compute_80 tmp.cu -lcudart -Xcompiler '-fPIC'

generate:
	g++ -shared -o main.so *.o -L/usr/local/cuda/lib64 -lcusparse -lcusparseLt
	g++ -shared -o tmp.so *.o -L/usr/local/cuda/lib64 -lcudart

# -fPIC 与位置无关进行编译

clean:
	rm -rf *.o *.so


.PHONY: generate clean