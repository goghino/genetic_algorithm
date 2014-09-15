#CPU specific configurations
CPUCC=g++
CPUCFLAGS=-g -O3

#GPU specific configurations
GPUCC=nvcc
GPUCFLAGS=-g -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
CUDA_DIR=$(shell dirname $(shell which nvcc))/..
CULIBS=-lcurand -lcudart -lnvToolsExt

#MPI/MPI+CUDA specific configurations
MPIBIN=~/forge/openmpi-1.8.2/install/bin
MPICC=$(MPIBIN)/mpic++
MPICU=$(MPIBIN)/mpic++
MPICU_RUN=$(MPIBIN)/mpirun

#absolute path to working dir
WORK_DIR=$(shell pwd)

all: cpu gpu mpi multi

generator: generator.c
	gcc -std=c99 $< -o $@

cpu: cpu_version.cpp generator
	$(CPUCC) $(CPUCFLAGS) $< -o $@
	
gpu: gpu_version.cu
	$(GPUCC) $(GPUCFLAGS) $< -o $@ -lcurand

mpi: mpi_gpu.o mpi_cpu.o
	$(MPICC) $(CPUCFLAGS) -L$(CUDA_DIR)/lib64 $^ -o $@ $(CULIBS)

mpi_gpu.o: mpi_version.cu mpi_version.h config.h
	$(GPUCC) $(GPUCFLAGS) -c $< -o $@

mpi_cpu.o: mpi_version.cpp mpi_version.h config.h
	$(MPICC) $(CPUCFLAGS) -c $< -o $@

multi: mpi_gpu_multi.o mpi_cpu_multi.o
	$(MPICU) $(CPUCFLAGS) -L$(CUDA_DIR)/lib64 $^ -o $@ $(CULIBS)

mpi_gpu_multi.o: mpi_version_multi.cu mpi_version_multi.h config.h
	$(GPUCC) $(GPUCFLAGS) -c $< -o $@

mpi_cpu_multi.o: mpi_version_multi.cpp mpi_version_multi.h config.h
	$(MPICU) $(CPUCFLAGS) -I$(CUDA_DIR)/include -c $< -o $@

#cuda aware launch of multi-gpu application
multirun:
	$(MPICU_RUN) -np 4 $(WORK_DIR)/multi $(WORK_DIR)/input.txt

analyze:
	amplxe-cl -collect hotspots -result-dir result $(MPICU_RUN) -n 4 $(WORK_DIR)/multi $(WORK_DIR)/input.txt

#$(MPICU_RUN) -np 4 amplxe-cl -r my_result --collect hotspots $(WORK_DIR)/multi $(WORK_DIR)/input.txt


#more like demonstration how to run executables than actual benchmark
run: 
	./generator 100 && ./cpu input.txt && ./gpu input.txt && $(MPICU_RUN) -np 3 $(WORK_DIR)/mpi $(WORK_DIR)/input.txt && $(MPICU_RUN) -np 4 $(WORK_DIR)/multi $(WORK_DIR)/input.txt && gnuplot plot.gnu

test:
	cuda-memcheck --leak-check full --report-api-errors yes ./gpu input.txt


#difference between Kepler and Fermi performance
bench:
	CUDA_VISIBLE_DEVICES=0 ./gpu input.txt && CUDA_VISIBLE_DEVICES=1 ./gpu input.txt

clean:
	rm -rf cpu gpu mpi *.o generator

