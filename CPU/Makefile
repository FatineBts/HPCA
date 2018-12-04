SRCS = neutron-cpu.c
EXE_NAME = neutron-cpu

CC = gcc
CFLAGS = -Wall -O3
LIBS = -lm -fopenmp

all: ${EXE_NAME}

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~

exec: 
	make
	./neutron-cpu 1.0 500000000 0.5 0.5 
