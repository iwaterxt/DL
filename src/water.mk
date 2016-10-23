# This file was generated using the following command:
# ./configure --shared

# Rules that enable valgrind debugging ("make valgrind")

valgrind: .valgrind

.valgrind:
	echo -n > valgrind.out
	for x in $(TESTFILES); do echo $$x>>valgrind.out; valgrind ./$$x >/dev/null 2>> valgrind.out; done
	! ( grep 'ERROR SUMMARY' valgrind.out | grep -v '0 errors' )
	! ( grep 'definitely lost' valgrind.out | grep -v -w 0 )
	rm valgrind.out
	touch .valgrind

CONFIGURE_VERSION := 1
OPENBLAS = /opt/OpenBLAS
OPENBLAS_VER = 0.2.14
OPENBLASLIBS = -L/opt/OpenBLAS/lib -lopenblas
OPENBLASLDFLAGS = -WI, -rpath=/opt/OpenBLAS/lib

# you have to make sure OPENBLAS is set.....

ifndef OPENBLAS
$(error OPENBLAS nor defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -pthread -fopenmp -Wno-sign-compare \
	-Wno-unused-function -Wno-reorder -std=c++0x \
	-I$(OPENBLAS)/include

LDFLAGS =  -rdynamic
LDLIBS = $(OPENBLAS) -lm -lpthread -ldl
CC = g++
CXX = g++
AR = ar
AS = as
RANDLIB = randlib

#Next section enables CUDA for compilation
CUDA = true
CUDATKDIR = /usr/local/cuda

CUDA_INCLUDE = -l$(CUDATKDIR)/include

CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --meachine 64 -DHAVE_CUDA

CXXFLAG += -DHAVE_CUDA -L$(CUATKDIR)/include

UNAME := $(shell uname)
#aware of fact in cuda60 there is no lib64, just lib.
ifeq ($(UNAME), Darwin)
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
else
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
endif
CUDA_LDLIBS += -lcublas -lcudart #LDLIBS : The libs are loaded later than static libs in implicit rule
