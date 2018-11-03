SRCS = neutron-hybride.cu
EXE_NAME = neutron-hybride

CC = nvcc -Xcompiler -fopenmp
CFLAGS =  
LIBS = -lgomp -lm -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcurand

all: ${EXE_NAME}

% : %.cu
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~