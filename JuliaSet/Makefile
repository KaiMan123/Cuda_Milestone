INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: julia_one julia_two

julia_one: julia_one.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc julia_one.cu -o julia_one $(INC) $(NVCCFLAGS) $(LIB)

julia_two: julia_two.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc julia_two.cu -o julia_two $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f julia_one julia_two
