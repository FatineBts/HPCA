SRCS = neutron-cpu.cu
EXE_NAME = neutron-cpu

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

exec: 
	./neutron-cpu 1.0 500000000 0.5 0.5 15000
