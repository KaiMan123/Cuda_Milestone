INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: blur_ls blur_cs blur_gs blur_lg blur_cg blur_gg

blur_ls: blur_ls.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_ls.cu -o blur_ls $(INC) $(NVCCFLAGS) $(LIB)

blur_cs: blur_cs.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_cs.cu -o blur_cs $(INC) $(NVCCFLAGS) $(LIB)

blur_gs: blur_gs.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_gs.cu -o blur_gs $(INC) $(NVCCFLAGS) $(LIB)

blur_lg: blur_lg.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_lg.cu -o blur_lg $(INC) $(NVCCFLAGS) $(LIB)

blur_cg: blur_cg.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_cg.cu -o blur_cg $(INC) $(NVCCFLAGS) $(LIB)

blur_gg: blur_gg.cu Makefile
	/usr/local/cuda-9.0/bin/nvcc blur_gg.cu -o blur_gg $(INC) $(NVCCFLAGS) $(LIB)


clean:
	rm -f blur_ls blur_cs blur_gs blur_lg blur_cg blur_gg
