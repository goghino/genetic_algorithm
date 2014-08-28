CPUCC=g++
CPUCFLAGS=
CPUEXECUTABLE=cpu

GPUCC=nvcc
GPUCFLAGS=-lcurand
GPUEXECUTABLE=gpu

all: mpi




mpi: mpi_version.cpp mpi_version.cu mpi_version.h
	nvcc -c mpi_version.cu -g -o mpi_gpu.o
	mpic++ -c mpi_version.cpp -g -o mpi_cpu.o
	mpic++ -L/opt/cuda/lib64 -stdlib=libstdc++ mpi_gpu.o mpi_cpu.o -o $@ -lcurand -lcudart

run: mpi
	mpirun -np 3 ./mpi input5.txt

clean:
	rm -rf cpu gpu generator mpi
	rm -rf *.o
