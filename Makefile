all:
	make -C CPU
	make -C GPU
	make -C GPU_CPU

clean:
	make clean -C CPU
	make clean -C GPU
	make clean -C GPU_CPU