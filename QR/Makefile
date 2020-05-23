CXX = nvcc
CULDFLAGS=-L/usr/local/cuda/lib
CFLAGS = -O2  -I $(CUDADIR)/include   -I /home/shaoshuai/LATER/include
LFLAGS = -L $(CUDADIR)/lib64  -lcublas -lcurand -lcudart -lcusparse  -lcuda  -Xlinker -lcusolver -lpthread -lm -ldl -lineinfo
CC = gcc


test: test.cu later_rhouqr.cu later_rgsqrf.cu later_bhouqr.cu
	$(CXX)  $(CFLAGS) $(LFLAGS) $^ -o $@