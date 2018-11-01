SRCS = neutron-seq.cu
EXE_NAME = neutron-seq

NVCC=nvcc  
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcurand

ifeq ($(DEBUGGING), y)
 CFLAGS=
 CUDA_FLAGS =
else
 CFLAGS=
 CUDA_FLAGS =  
endif 

all: ${EXE_NAME}

% : %.cu
$(NVCC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIB)

clean:
rm -f ${EXE_NAME} *.o *~
