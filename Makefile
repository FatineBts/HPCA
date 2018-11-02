all:
	make -C GPU
	make -C CPU

clean:
	make clean -C GPU
	make clean -C CPU
