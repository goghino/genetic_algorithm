CPUCC=g++
CPUCFLAGS=

GPUCC=nvcc
GPUCFLAGS=

MPICC=mpic++
MPICU=/home/jkardos/openmpi-1.8.1/install/bin/mpic++


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

multi: mpi_version_multi.cu mpi_version_multi.cpp mpi_version_multi.h
	$(GPUCC) $(GPUCFLAGS) -c mpi_version_multi.cu -o mpi_gpu_multi.o
	$(MPICU) $(CPUCFLAGS) -I/opt/cuda/include -c mpi_version_multi.cpp -o mpi_cpu_multi.o
	$(MPICU) $(CPUCFLAGS) -L/opt/cuda/lib64 -stdlib=libstdc++ mpi_gpu_multi.o mpi_cpu_multi.o -o $@ -lcurand -lcudart


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
