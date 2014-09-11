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
#include <thrust/device_ptr.h>

#include "mpi_version.h"

using namespace std;

#define maxGenerationNumber 1500
#define maxConstIter 100
#define targetErr (N_POINTS*0.005)
#define mu_individuals 0.5
#define sigma_individuals 0.66
#define mu_genes 0.56
#define sigma_genes 0.75

#define THREAD 128
#define BLOCK (POPULATION_SIZE/THREAD)


/**
    An individual fitness function is the difference between measured f(x) and
    approximated polynomial g(x), built using individual's coeficients,
    evaluated on input data points.

    Smaller value means bigger fitness
*/
__global__ void fitness_evaluate(float *individuals, float *points, float *fitness)
{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= POPULATION_SIZE)
        return;

    float sumError = 0;

    //for every given data point
	for(int pt=0; pt<N_POINTS; pt++)
	{
		float f_approx = 0.;
		
        //for every polynomial parameter: Ci * x^(order)
		for (int order=0; order < INDIVIDUAL_LEN; order++)
		{
			f_approx += individuals[idx + order*POPULATION_SIZE] * pow(points[pt], order);
		}

		sumError += pow(f_approx - points[N_POINTS+pt], 2);
	}
	
    //The lower value of fitness is, the better individual fits the model
	fitness[idx] = sumError;
}


/**
    Individual is set of coeficients c1-c4. 

    For example:
    parent1 == [0 0 0 0]
    parent2 == [1 1 1 1]
    crosspoint(random between 1 and 3) = 2
      then
    child1  = [0 0 1 1]
    child2  = [1 1 0 0]
*/

__global__ void crossover(float *population_dev, curandState *state)
{
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //Replace only second half of the population by new individuals
    //created by crossover from the first half of the population
    if(idx >= POPULATION_SIZE || idx<POPULATION_SIZE/2)
        return;
   
    //randomly select two fit parrents for mating from the fittest half of the population
    curandState localState = state[idx];
	int parent1_i = (curand(&localState) % (POPULATION_SIZE/2)) * INDIVIDUAL_LEN;
	int parent2_i = (curand(&localState) % (POPULATION_SIZE/2)) * INDIVIDUAL_LEN;


    //select crosspoint, do not select beginning and end of individual as crosspoint
	int crosspoint = curand(&localState) % (INDIVIDUAL_LEN - 2) + 1 ;
	state[idx] = localState;

    //do actual crossover
    for(int j=0; j<INDIVIDUAL_LEN; j++){
        if(j<crosspoint)
        {
            population_dev[idx +j*POPULATION_SIZE]
                = population_dev[parent1_i + j*POPULATION_SIZE];
            population_dev[idx + j*POPULATION_SIZE + 1]
                = population_dev[parent2_i + j*POPULATION_SIZE];  
        } else
        {
            population_dev[idx + j*POPULATION_SIZE]
                = population_dev[parent2_i + j*POPULATION_SIZE];
            population_dev[idx + j*POPULATION_SIZE + 1]
                = population_dev[parent1_i + j*POPULATION_SIZE];
        }
    }

}

/**
    Generates probabilities for mutation of individuals and their genes
    into arrays in device global memory
*/
void generateMutProbab(float** mutIndivid, float **mutGene, curandGenerator_t generator)
{
    //mutation rate of individuals
    curandGenerateNormal(generator, *mutIndivid,
                        POPULATION_SIZE, mu_individuals, sigma_individuals);
    check_cuda_error("Error in normalGenerating 1");

    //mutation rate of each gene
    curandGenerateNormal(generator, *mutGene,
                        POPULATION_SIZE*INDIVIDUAL_LEN, mu_genes, sigma_genes);
    check_cuda_error("Error in normalGenerating 2");
}

/**
    Mutation is addition of noise to genes, given mean and stddev.

    probabilities of mutating individuals and their 
    genes is computed before calling this kernel
    @mutGene
    @mutIndivid

    For example(binary representation of genes):
    individual == [1 1 1 1]
    mutNumber = 2
    loop 2 times:
       1st: num_of_bit_to_mutate = 2
            inverse individuals[2]   ->   [1 1 0 1] 
       2nd: num_of_bit_to_mutate = 0
            inverse individuals[0]   ->   [0 1 0 1]
    return mutated individual         [0 1 0 1]
*/
__global__ void mutation(float *individuals, curandState *state,
                         float* mutIndivid, float* mutGene)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    curandState localState = state[idx];
    
    //first individual is not mutated to keep the best solution unchanged    
    if(idx >= POPULATION_SIZE || idx < 1)
        return;

    float mutationRate = mutIndivid[idx];

    for(int j=0; j<INDIVIDUAL_LEN; j++)
    {
        int flip_idx = idx + j*POPULATION_SIZE;
        //probability of mutating gene 
        if(mutGene[flip_idx] < mutationRate) {
            individuals[flip_idx] += 0.01*(2*curand_uniform(&localState)-1);
        } 
    }

    state[idx] = localState;
}

/**
    Sets up indexes for thrust::sort_by_key
*/
__global__ void setIndexes(int *indexes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx>=POPULATION_SIZE) 
        return;

    indexes[idx] = idx;
}


/*
    population - sorted individuals according to their fitness

	individuals with small (good) fitness value are put to the beginning 
	individuals with large (bad) fitness value are placed at the end;
*/
__global__ void selection(float *population, float *newPopulation, int* indexes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //only first half needs to be placed in sorted manner
    //second half will be overwritten anyway
    if(idx > POPULATION_SIZE/2)
        return;

    //reorder population so that fittest individuals are first
    for (int j=0; j<INDIVIDUAL_LEN; j++)
    {
        newPopulation[idx + j*POPULATION_SIZE]
            = population[indexes[idx] + j*POPULATION_SIZE];
    }
}

/**
    Initializes seed for CUDA random generator
*/
__global__ void initCurand(curandState *state)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(6482, id, 0, &state[id]);
}


/**
    Initializes initial population by random values. Use range <-5.0, 5.0>
*/
__global__ void initPopulation(float *population, curandState *state)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[id];

    if(id < POPULATION_SIZE)
    {
        for(int i=0; i<INDIVIDUAL_LEN; i++)
            population[id + i*POPULATION_SIZE] = 10*curand_uniform(&localState) - 5;        
    }
}

//------------------------------------------------------------------------------


/*
    ------------------------
    | Main body of the GA  |
    ------------------------

    Computes approximation of given points
*/
void computeGA(float *points, int deviceID,
               float *solution, float *bestFitness_o, int *genNumber_o, double *time_o)
{

    cudaSetDevice(deviceID);
    check_cuda_error("Setting device");

    /**
        Allocations of memory
    */
    //device memory for holding input points
    float *points_dev;
    cudaMalloc(&points_dev, 2*N_POINTS*sizeof(float)); // [x, f(x)+err]
    check_cuda_error("Error allocating device memory");
    cudaMemcpy(points_dev, points, 2*N_POINTS*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error("Error copying data");

    //arrays to hold old and new population    
    float *population_dev;
    cudaMalloc(&population_dev, POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(float));
    check_cuda_error("Error allocating device memory");

    float *newPopulation_dev;
    cudaMalloc(&newPopulation_dev, POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(float));
    check_cuda_error("Error allocating device memory");

    //arrays that keeps fitness of individuals withing current population
    float *fitness_dev;
    cudaMalloc(&fitness_dev, POPULATION_SIZE*sizeof(float));
    check_cuda_error("Error allocating device memory");

    //key value for sorting
    int *indexes_dev;
    cudaMalloc(&indexes_dev, POPULATION_SIZE*sizeof(int));
    check_cuda_error("Error allocating device memory");

    curandState *state_random;
    cudaMalloc((void **)&state_random,POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(curandState));
    check_cuda_error("Allocating memory for curandState");

    //mutation probabilities
    float* mutIndivid_d;
    cudaMalloc((void **) &mutIndivid_d,POPULATION_SIZE*sizeof(float));
    check_cuda_error("Allocating memory in mutIndivid_d");

    float* mutGene_d;
    cudaMalloc((void **)&mutGene_d,POPULATION_SIZE*INDIVIDUAL_LEN*sizeof(float));
    check_cuda_error("Allocating memory in mutGene_d");

    //create PRNG for generating mutation probabilities
    curandGenerator_t generator;

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    check_cuda_error("Error in curandCreateGenerator");

    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
    check_cuda_error("Error in curandSeed");

    //recast device pointers into thrust copatible pointers
    thrust::device_ptr<int> indexes_thrust = thrust::device_pointer_cast(indexes_dev);
    thrust::device_ptr<float> fitnesses_thrust = thrust::device_pointer_cast(fitness_dev);


    //Initialize first population (with zeros or some random values)
    initCurand<<<BLOCK, THREAD>>>(state_random);
    initPopulation<<<BLOCK, THREAD>>>(population_dev, state_random); //<-5, 5>

    /**
        Main GA loop
    */
    int t1 = clock(); //start timer

    int generationNumber = 0;
    int noChangeIter = 0;

    float bestFitness = INFINITY;
    float previousBestFitness = INFINITY;

	while ( (generationNumber < maxGenerationNumber)
            && (bestFitness > targetErr)
            && (noChangeIter < maxConstIter) )
	{
		generationNumber++;
	
        /** crossover first half of the population and create new population */
		crossover<<<BLOCK,THREAD>>>(population_dev, state_random);
        cudaDeviceSynchronize();

		/** mutate population and childrens in the whole population*/
        generateMutProbab(&mutIndivid_d, &mutGene_d, generator);
		mutation<<<BLOCK,THREAD>>>(population_dev, state_random, mutIndivid_d, mutGene_d);
        cudaDeviceSynchronize();
		
        /** evaluate fitness of individuals in population */
		fitness_evaluate<<<BLOCK,THREAD>>>(population_dev, points_dev, fitness_dev);
        cudaDeviceSynchronize();
        
        /** select individuals for mating to create the next generation,
            i.e. sort population according to its fitness and keep
            fittest individuals first in population  */
        setIndexes<<<BLOCK,THREAD>>>(indexes_dev);
        cudaDeviceSynchronize();

        thrust::sort_by_key(fitnesses_thrust, fitnesses_thrust+POPULATION_SIZE, indexes_thrust);

        selection<<<BLOCK,THREAD>>>(population_dev, newPopulation_dev, indexes_dev);
        cudaDeviceSynchronize();
        
        //swap populations        
        float *tmp = population_dev;
        population_dev = newPopulation_dev;
        newPopulation_dev = tmp;
        

        /** time step evaluation - convergence criterion check */
        //get BEST FITNESS to host
        cudaMemcpy(&bestFitness, fitness_dev, sizeof(float), cudaMemcpyDeviceToHost);
        check_cuda_error("Coping fitnesses_dev[0] to host");
        
        //check if the fitness is decreasing or if we are stuck at local minima
        if(fabs(bestFitness - previousBestFitness) < 0.01f)
            noChangeIter++;
        else
            noChangeIter = 0;
        previousBestFitness = bestFitness;

        //log message
        #if defined(DEBUG)
        cout << "#" << generationNumber<< " Fitness: " << bestFitness << \
        " Iterations without change: " << noChangeIter << endl;
        #endif
	}

    int t2 = clock(); //stop timer

    /**
        Results
    */

    //get solution from device to host
    for(int i=0; i<INDIVIDUAL_LEN; i++){
        cudaMemcpy(&solution[i], &population_dev[i*POPULATION_SIZE],
                   sizeof(float), cudaMemcpyDeviceToHost);
        check_cuda_error("Coping fitnesses_dev[0] to host");
    }

    *bestFitness_o = bestFitness;
    *genNumber_o = generationNumber;
    *time_o = (t2-t1)/(double)CLOCKS_PER_SEC;



    /**
        Free memory
    */
    cudaFree(points_dev);//input points
    cudaFree(fitness_dev);//fitness array
    cudaFree(indexes_dev);//key for sorting
    cudaFree(population_dev);
    cudaFree(newPopulation_dev);
    cudaFree(state_random);//state curand
    cudaFree(mutIndivid_d);//mutation probability
    cudaFree(mutGene_d);//mutation probability

    curandDestroyGenerator(generator);

    cudaDeviceReset();
    check_cuda_error("Resseting device");
}

//------------------------------------------------------------------------------

void check_cuda_error(const char *message)
{
        cudaError_t err = cudaGetLastError();
            if (err!=cudaSuccess){
             printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
             exit(1);
            }
}
