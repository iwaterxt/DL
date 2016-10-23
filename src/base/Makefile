CC = g++

CXX_OPTION_FLAGS = -g -msse -msse2 -Wall  -fopenmp -Wno-sign-compare -Wno-unused-function -Wno-reorder -rdynamic -std=c++0x


object = nnet-io.o

libio.a: $(object)
	ar cru libio.a $(object)
	ranlib libio.a
	chmod 775 libio.a

	
nnet-io.o:nnet-io.cc
	$(CC) $(CXX_OPTION_FLAGS) -o nnet-io.o -c nnet-io.cc

.PHONY:clean

clean :
	rm -rf *.o .a