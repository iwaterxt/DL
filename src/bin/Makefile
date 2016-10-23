GENCODE_FLAGS = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 

NVCC_OPTION_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64  $(GENCODE_FLAGS)

CXX_OPTION_FLAGS = -g -msse -msse2 -Wall -fPIC -fopenmp -Wno-sign-compare -Wno-unused-function -Wno-reorder -rdynamic -std=c++0x 

INCLUDE = -I/usr/local/cuda/include -I/opt/OpenBLAS/include

LIB = -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda -lcurand -lcudnn  -L/opt/OpenBLAS/lib -lopenblas -lpthread  -lm 

AddLIB = ../matrix/libMatrix.a  ../cumatrix/libcuMatrix.a ../nnet/libnnet.a ../base/libio.a

NV_CXX = /usr/local/cuda/bin/nvcc 

CXX = g++

all:smart-network-testing smart-network-training smart-network-initial

object = smart-network-training.o smart-network-testing.o smart-network-training.o
 
smart-network-testing:smart-network-testing.o $(AddLIB)
	$(NV_CXX)  $(NVCC_OPTION_FLAGS) $(INCLUDE)   smart-network-testing.o -o smart-network-testing  $(AddLIB) $(LIB)

smart-network-training:smart-network-training.o $(AddLIB)
	$(NV_CXX)  $(NVCC_OPTION_FLAGS) $(INCLUDE)  smart-network-training.o -o smart-network-training  $(AddLIB) $(LIB)

smart-network-initial:smart-network-initial.o $(AddLIB)
	$(NV_CXX)  $(NVCC_OPTION_FLAGS) $(INCLUDE)  smart-network-initial.o -o smart-network-initial  $(AddLIB) $(LIB)

smart-network-training.o:smart-network-training.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o smart-network-training.o -c smart-network-training.cc

smart-network-testing.o:smart-network-testing.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o smart-network-testing.o -c smart-network-testing.cc

smart-network-initial.o:smart-network-initial.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o smart-network-initial.o -c smart-network-initial.cc

.PHONY:clean
 
clean :
	rm -rf *.o smart-network-testing smart-network-training smart-network-initial
