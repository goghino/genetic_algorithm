CPUCC=g++
CPUCFLAGS=

GPUCC=nvcc
GPUCFLAGS=

MPICC=mpic++


all: cpu gpu mpi

cpu: cpu_version.cpp
	$(CPUCC) $(CPUCFLAGS) $< -o $@
	gcc -std=c99 generator.c -o generator
	
gpu: gpu_version.cu
	$(GPUCC) $(GPUCFLAGS) $< -o $@ -lcurand

mpi: mpi_version.cpp mpi_version.cu mpi_version.h
	$(GPUCC) $(GPUCFLAGS) -c mpi_version.cu -o mpi_gpu.o
	$(MPICC) $(CPUCFLAGS) -c mpi_version.cpp -o mpi_cpu.o
	$(MPICC) $(CPUCFLAGS) -L/opt/cuda/lib64 -stdlib=libstdc++ mpi_gpu.o mpi_cpu.o -o $@ -lcurand -lcudart

run: 
	./generator 100    #generate input file
	./cpu input.txt
	./gpu input.txt
	mpirun -np 3 ./mpi input5.txt
	gnuplot plot.gnu   #need to set found polynomial parameters manually

test:
	cuda-memcheck --leak-check full --report-api-errors yes ./gpu input.txt

bench:
	CUDA_VISIBLE_DEVICES=0 ./gpu input.txt
	CUDA_VISIBLE_DEVICES=1 ./gpu input.txt

clean:
	rm -rf cpu gpu mpi *.o generator
