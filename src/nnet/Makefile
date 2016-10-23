GENCODE_FLAGS =  -gencode arch=compute_50,code=sm_50 -gencode arch=compute_30,code=sm_30

NVCC_OPTION_FLAGS = -g -Xcompiler  --verbose --machine 64  $(GENCODE_FLAGS)

CXX_OPTION_FLAGS = -g -msse -msse2 -Wall   -fopenmp -Wno-sign-compare -Wno-unused-function -Wno-reorder -rdynamic -std=c++0x 

INCLUDE = -I/usr/local/cuda/include  -I/opt/OpenBLAS/include 

LIB = -L/usr/local/cuda/lib64 -lcublas -lcudart  -lcuda -lcurand -lcudnn -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -lm 


NV_CXX = /usr/local/cuda/bin/nvcc 

CXX = g++

object = nnet-nnet.o nnet-loss.o nnet-component.o


libnnet.a: $(object)
	ar cru libnnet.a $(object)
	ranlib libnnet.a
	chmod 775 libnnet.a

all:$(object)

nnet-nnet.o:nnet-nnet.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o nnet-nnet.o -c nnet-nnet.cc

nnet-loss.o:nnet-loss.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o nnet-loss.o -c nnet-loss.cc 

nnet-component.o:nnet-component.cc
	$(CXX) $(CXX_OPTION_FLAGS) $(INCLUDE) -o nnet-component.o -c nnet-component.cc

.PHONY:clean

clean :
	rm -rf *.o *.a