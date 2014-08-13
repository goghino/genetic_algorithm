CPUCC=g++
CPUCFLAGS=
CPUEXECUTABLE=cpu

GPUCC=nvcc
GPUCFLAGS=-lcurand
GPUEXECUTABLE=gpu

all: cpu gpu generator mpi




cpu: cpu_version.cpp
	g++ $< -g -o $(CPUEXECUTABLE)    

gpu: gpu_version.cu
	nvcc $< -arch=sm_20 -g -o $(GPUEXECUTABLE) -lcurand

mpi: mpi_version.cpp mpi_version.cu mpi_version.h
	nvcc -c mpi_version.cu -g -o mpi_gpu.o
	mpic++ -c mpi_version.cpp -g -o mpi_cpu.o
	mpic++ -L/opt/cuda/lib64 -stdlib=libstdc++ mpi_gpu.o mpi_cpu.o -o $@ -lcurand -lcudart

generator: generator.c
	gcc -std=c99 generator.c -o generator

run: 
	./generator 100    #generate input file
	./$(CPUEXECUTABLE) input.txt
	./$(GPUEXECUTABLE) input.txt
	gnuplot plot.gnu   #need to set found polynomial parameters manually

clean:
	rm -rf cpu gpu generator mpi
	rm -rf *.o
