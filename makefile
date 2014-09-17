#CPU specific configurations
CPUCC=g++
CPUCFLAGS=

#GPU specific configurations
GPUCC=nvcc
GPUCFLAGS=
CUDA_DIR=/opt/cuda
CULIBS=-lcurand -lcudart -lnvToolsExt

#MPI/MPI+CUDA specific configurations
MPICC=mpic++
MPICU=/home/jkardos/openmpi-1.8.1/install/bin/mpic++
MPICU_RUN=/home/jkardos/openmpi-1.8.1/install/bin/mpirun

#absolute path to working dir
WORK_DIR=/home/jkardos/certification/genetic_algorithm/code


######################## Build rules ############################################ 

all: cpu gpu mpi

cpu: cpu_version.cpp
	$(CPUCC) $(CPUCFLAGS) $< -o $@
	gcc -std=c99 generator.c -o generator


gpu: gpu_version.cu
	$(GPUCC) $(GPUCFLAGS) $< -o $@ -lcurand


mpi: mpi_version.cpp mpi_version.cu mpi_version.h
	$(GPUCC) $(GPUCFLAGS) -c mpi_version.cu -o mpi_gpu.o
	$(MPICC) $(CPUCFLAGS) -c mpi_version.cpp -o mpi_cpu.o
	$(MPICC) $(CPUCFLAGS) -L$(CUDA_DIR)/lib64 mpi_gpu.o mpi_cpu.o -o $@ $(CULIBS)

multi: mpi_version_multi.cu mpi_version_multi.cpp mpi_version_multi.h
	$(GPUCC) $(GPUCFLAGS) -c mpi_version_multi.cu -o mpi_gpu_multi.o
	$(MPICU) $(CPUCFLAGS) -I$(CUDA_DIR)/include -c mpi_version_multi.cpp -o mpi_cpu_multi.o
	$(MPICU) $(CPUCFLAGS) -L$(CUDA_DIR)/lib64 mpi_gpu_multi.o mpi_cpu_multi.o -o $@ $(CULIBS)


#########################  Commands to run separate versions ######################## 
rungpu:
	./gpu input.txt
	
runcpu:
	./cpu input.txt


runmpi:
	mpirun -n 4 ./mpi input.txt

#cuda aware launch of multi-gpu application
runmultimpi:
	$(MPICU_RUN) -np 4 $(WORK_DIR)/multi $(WORK_DIR)/input.txt

######################## Additional make rules ################################

analyze:
	amplxe-cl -collect hotspots -result-dir result $(MPICU_RUN) -n 4 $(WORK_DIR)/multi $(WORK_DIR)/input.txt

#$(MPICU_RUN) -np 4 amplxe-cl -r my_result --collect hotspots $(WORK_DIR)/multi $(WORK_DIR)/input.txt


#more like demonstration how to run executables than actual benchmark
run: 
	./generator 100    #generate input file
	./cpu input.txt
	./gpu input.txt
	mpirun -np 3 ./mpi input5.txt
	multirun
	gnuplot plot.gnu   #need to set found polynomial parameters manually

test:
	cuda-memcheck --leak-check full --report-api-errors yes ./gpu input.txt
	cuda-memcheck --leak-check full ./gpu input.txt


#difference between Kepler and Fermi performance
bench:
	CUDA_VISIBLE_DEVICES=0 ./gpu input.txt
	CUDA_VISIBLE_DEVICES=1 ./gpu input.txt

clean:
	rm -rf cpu gpu mpi *.o generator multi
