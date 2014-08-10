CPUCC=g++
CPUCFLAGS=
CPUSOURCES=cpu_version.cpp
CPUEXECUTABLE=cpu

GPUCC=nvcc
GPUCFLAGS=-lcurand
GPUSOURCES=gpu_version.cu
GPUEXECUTABLE=gpu

all:
	g++ $(CPUSOURCES) -g -o $(CPUEXECUTABLE)
	nvcc $(GPUSOURCES) -arch=sm_20 -g -o $(GPUEXECUTABLE) -lcurand
	gcc -std=c99 generator.c -o generator
run: 
	./generator 100    #generate input file
	./$(CPUEXECUTABLE) input.txt
	./$(GPUEXECUTABLE) input.txt
	gnuplot plot.gnu   #need to set found polynomial parameters manually

clean:
	rm -rf $(GPUEXECUTABLE)
	rm -rf $(CPUEXECUTABLE)
