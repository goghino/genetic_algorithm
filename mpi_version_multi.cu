/**

Genetic algorithm for finding function aproximation. GPU accelerated version

Given data points {x, f(x)+noise} generated by noisy polynomial function
f(x) = c3*x^3 + c2*x^2 + c1*x + c0,
find unknown parameters c1, c2, c3 and c0.


Inputs:
• The set of points on a surface (500–1000);
• The size of population P (1000–2000);
• E_m , D_m – mean and variance for Mutation to generate the random number of mutated genes;
• maxIter - the maximum number of generations, 
  maxConstIter - the maximum number of generations with constant value of the best fitness.

Outputs:
• The time of processing on GPU;
• The set of coefficients of the polynomial that approximates the given set of points;
• The best fitness value;
• The last generation number (number of evaluated iterations).
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <algorithm>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>

#include "mpi_version_multi.h"

#include "config.h"
#include "kernels.h"
#include "nvToolsExt.h"

using namespace std;

#define THREAD 128
#define BLOCK (POPULATION_SIZE/THREAD)

// Override cudaMalloc with our function call so that thrust::sort_by_key
// does not allocate/free working memory every iteration
extern __thread bool cudaMallocReuse;

void check_cuda_error(const char *message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
		exit(1);
	}
}


//------------------------------------------------------------------------------
//                              GPU KERNELS
//------------------------------------------------------------------------------

/*
    Transforms population matrix from coalesced pattern
    to continuous array of individuals

    @size - count of individuals in population
*/
__global__ void transpose(float *population_devT, float *population_dev, int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    for(int i = 0; i<INDIVIDUAL_LEN; i++)
    {   
        population_devT[idx*INDIVIDUAL_LEN + i] = population_dev[i*size + idx];
    }
}

/*
    Transforms population matrix from continous pattern
    to coalesced arrangement in memory

    @size - count of individuals in population
*/
__global__ void transpose_inverse(float *population_devT, float *population_dev, int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    for(int i = 0; i<INDIVIDUAL_LEN; i++)
    {   
        population_devT[i*size + idx] = population_dev[idx*INDIVIDUAL_LEN + i];
    }
}

//------------------------------------------------------------------------------
//                 Encapsulating GPU functions for kernel calls
//------------------------------------------------------------------------------

/*
    Initialize population with random values. All workload is done on
    master process.
*/
void doInitPopulation(float *population_dev, curandState *state_random)
{
    int block = POPULATION_SIZE/THREAD;
    initCurand<<<block, THREAD>>>(state_random);
    initPopulation<<<block, THREAD>>>(population_dev, state_random); //<-5, 5>
    cudaDeviceSynchronize();
}

/*
    Initializes only curandState array of size @size for slave processes
*/
void doInitCurandOnly(curandState *state_random, int size)
{
    int block = size/THREAD;
    initCurand<<<block, THREAD>>>(state_random);
}

/**
    Crossover population, all workload is done on master process.
*/
void doCrossover(float *population_dev, curandState* state_random)
{
    int block = POPULATION_SIZE/THREAD;
    crossover<<<block,THREAD>>>(population_dev, state_random);
    cudaDeviceSynchronize();
}

/**
    Mutate population, performed on local portion of population on each process
    @size - number of individuals to process
*/
void doMutation(float *population_dev, curandState *state_random, int size,
                float *mutIndivid_d, float *mutGene_d, curandGenerator_t generator)
{
    generateMutProbab(&mutIndivid_d, &mutGene_d, generator, size);
    cudaDeviceSynchronize();

    //TODO size%thread != 0
    int block = size/THREAD;
    mutation<<<block,THREAD>>>(population_dev, state_random,
                                mutIndivid_d, mutGene_d, size);
    cudaDeviceSynchronize();
		
}		

/**
    Evaluate fitness of each individual from population,
    performed on local portion of population on each process

    @size - number of individuals to process
*/
void doFitness_evaluate(float *population_dev, float *points_dev, float *fitness_dev,
                        int size)
{
    int block = size/THREAD;

    fitness_evaluate<<<block,THREAD>>>(population_dev, points_dev, fitness_dev, size);
    cudaDeviceSynchronize();
}

/**
    Sort individuals according to fitness value
*/
void doSelection(thrust::device_ptr<float>fitnesses_thrust,
                 thrust::device_ptr<int>indexes_thrust, int *indexes_dev,
                 float *population_dev, float* newPopulation_dev)
{
    int block = POPULATION_SIZE/THREAD;

    setIndexes<<<block,THREAD>>>(indexes_dev);
    cudaDeviceSynchronize();


    nvtxRangePushA("thrust::sort");

#ifdef THRUST_REUSE_MALLOC
	cudaMallocReuse = true;
#endif

    //sort fitness array
    thrust::sort_by_key(fitnesses_thrust, fitnesses_thrust+POPULATION_SIZE, indexes_thrust);

#ifdef THRUST_REUSE_MALLOC
	cudaMallocReuse = false;
#endif

    nvtxRangePop();

    //reorder population according to fitness values
    selection<<<block,THREAD>>>(population_dev, newPopulation_dev, indexes_dev);
    cudaDeviceSynchronize();
}

/*
    Transforms population matrix from coalesced pattern
    to continuous array of individuals

    @size - count of individuals in population
*/
void doTranspose(float *population_devT, float *population_dev, int size)
{
    int block = size/THREAD;

    transpose<<<block,THREAD>>>(population_devT, population_dev, size);
}

/*
    Transforms population matrix from continous pattern
    to coalesced arrangement in memory

    @size - count of individuals in population
*/
void doTranspose_inverse(float *population_devT, float *population_dev, int size)
{
    int block = size/THREAD;

    transpose_inverse<<<block,THREAD>>>(population_devT, population_dev, size);
}
