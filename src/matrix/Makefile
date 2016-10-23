CC = g++

CC_OPTIONS = -g -msse -msse2 -Wall  -fopenmp -Wno-sign-compare -Wno-unused-function -Wno-reorder -rdynamic -std=c++0x

LIBS = -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -lm -L/usr/local/cuda/lib64 

INCLUDE = -I/opt/OpenBLAS/include -I/usr/local/cuda/include

object = Vector.o Matrix.o 

all: Matrix-test libMatrix.a

libMatrix.a: $(object)
	ar cru libMatrix.a $(object)
	ranlib libMatrix.a
	chmod 775 libMatrix.a

Matrix-test:$(object) Matrix-test.o
	$(CC) $(CC_OPTIONS)  $(object) Matrix-test.o -o Matrix-test  $(INCLUDE) $(LIBS)

Matrix-test.o:Matrix-test.cc
	$(CC) $(CC_OPTIONS)  $(INCLUDE) -o Matrix-test.o -c Matrix-test.cc
	
Vector.o:Vector.cc
	$(CC) $(CC_OPTIONS)  $(INCLUDE) -o Vector.o -c Vector.cc

Matrix.o:Matrix.cc
	$(CC) $(CC_OPTIONS)  $(INCLUDE) -o Matrix.o -c Matrix.cc 



.PHONY:clean

clean :
	rm -rf *.o .a
