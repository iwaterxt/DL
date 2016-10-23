GENCODE_FLAGS =-gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 

NVCC_OPTION_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64  $(GENCODE_FLAGS)

CXX_OPTION_FLAGS = -g -msse -msse2 -Wall -fPIC -fopenmp -Wno-sign-compare -Wno-unused-function -Wno-reorder -rdynamic -std=c++0x 

INCLUDE = -I/usr/local/cuda/include  -I/opt/OpenBLAS/include

LIB = -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda -lcurand -lcudnn -L/opt/OpenBLAS/lib -lopenblas -lpthread  -lm 

AddLIB = ../matrix/libMatrix.a ../base/libio.a

NV_CXX = /usr/local/cuda/bin/nvcc 

CXX = g++

object = cuMatrix.o cuVector.o cuKernel.o 

all:cu-test libcuMatrix.a
#
libcuMatrix.a: $(object)
	ar cru libcuMatrix.a $(object)
	ranlib libcuMatrix.a
	chmod 775 libcuMatrix.a

cu-test:$(object) cu-test.o
	$(NV_CXX)  $(NVCC_OPTION_FLAGS)   $(INCLUDE) $(object)  cu-test.o -o cu-test  $(AddLIB)  $(LIB)

cu-test.o: cu-test.cc 
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o cu-test.o -c cu-test.cc

cuMatrix.o: cuMatrix.cc cuKernel.cu 
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o cuMatrix.o -c cuMatrix.cc

cuVector.o: cuVector.cc cuKernel.cu 
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o cuVector.o -c cuVector.cc

cuKernel.o: cuKernel.cu 
	$(NV_CXX) $(NVCC_OPTION_FLAGS)  $(INCLUDE) -o cuKernel.o -c cuKernel.cu 

.PHONY:clean

clean:
	rm *.o cu-test *.a
