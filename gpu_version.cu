/**

Genetic algorithm for finding function aproximation. GPU accelerated version

Given data points {x, f(x)+noise} generated by noisy polynomial function
f(x) = c4*x^3 + c3*x^2 + c2*x + c1,
find unknown parameters c1, c2, c3 and c4.


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

#include <iterator>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "config.h"
#include "check.h"
#include "kernels.h"

void check_cuda_error(const char *message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
		exit(1);
	}
}

using namespace std;

#define THREAD 128
#define BLOCK (POPULATION_SIZE / THREAD)

// Reads input file with noisy points. Points will be approximated by 
// polynomial function using GA.
static float *readData(const char *name, int *N_POINTS);

// Override cudaMalloc with our function call so that thrust::sort_by_key
// does not allocate/free working memory every iteration
extern __thread bool cudaMallocReuse;

/*
    ------------------------
    | Main body of the GA  |
    ------------------------
*/
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " inputFile" << endl;    
        return -1;
    }

    //read input data - points to approximate by a polynomial
    int N_POINTS;
    float *points = readData(argv[1], &N_POINTS);

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
    cudaMemset(newPopulation_dev, 0, POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(float));
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

    curandSetPseudoRandomGeneratorSeed(generator, 0);
    check_cuda_error("Error in curandSeed");

    //recast device pointers into thrust copatible pointers
    thrust::device_ptr<int> indexes_thrust = thrust::device_pointer_cast(indexes_dev);
    thrust::device_ptr<float> fitnesses_thrust = thrust::device_pointer_cast(fitness_dev);


    //Initialize first population (with zeros or some random values)
    initCurand<<<BLOCK, THREAD>>>(state_random);
    initPopulation<<<BLOCK, THREAD>>>(population_dev, state_random); //<-5, 5>
    cudaDeviceSynchronize();

    /**
        Main GA loop
    */
    int t1 = clock(); //start timer

    int generationNumber = 0;
    int noChangeIter = 0;

    float bestFitness = INFINITY;
    float previousBestFitness = INFINITY;

#ifdef PERF_METRIC
    float *t_fitness = new float[maxGenerationNumber];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int TOTAL_POINTS = N_POINTS*POPULATION_SIZE;
#endif

	while ( (generationNumber < maxGenerationNumber)
            /*&& (bestFitness > targetErr)
            && (noChangeIter < maxConstIter) */)
	{
		generationNumber++;
	
        /** crossover first half of the population and create new population */
		crossover<<<BLOCK, THREAD>>>(population_dev, state_random);
        cudaDeviceSynchronize();

		/** mutate population and childrens in the whole population*/
        generateMutProbab(&mutIndivid_d, &mutGene_d, generator, POPULATION_SIZE);
        cudaDeviceSynchronize();
		mutation<<<BLOCK, THREAD>>>(population_dev, state_random,
                                    mutIndivid_d, mutGene_d, POPULATION_SIZE);
        cudaDeviceSynchronize();

#ifdef PERF_METRIC		
    cudaEventRecord(start, 0);
#endif

        /** evaluate fitness of individuals in population */
		fitness_evaluate<<<BLOCK, THREAD>>>(population_dev,
                                            points_dev, N_POINTS,
                                            fitness_dev, POPULATION_SIZE);
        cudaDeviceSynchronize();

#ifdef PERF_METRIC
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(t_fitness[generationNumber-1]), start, stop);
    
    //total points in generation / kernel execution time 
    t_fitness[generationNumber-1] = TOTAL_POINTS/t_fitness[generationNumber-1]/1000000;     
#endif

        /** select individuals for mating to create the next generation,
            i.e. sort population according to its fitness and keep
            fittest individuals first in population  */
        setIndexes<<<BLOCK, THREAD>>>(indexes_dev);
        cudaDeviceSynchronize();

#ifdef THRUST_REUSE_MALLOC
	cudaMallocReuse = true;
#endif

        thrust::stable_sort_by_key(fitnesses_thrust, fitnesses_thrust + POPULATION_SIZE, indexes_thrust);

#ifdef THRUST_REUSE_MALLOC
	cudaMallocReuse = false;
#endif

        selection<<<BLOCK, THREAD>>>(population_dev, newPopulation_dev, indexes_dev);
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
        if(fabs(bestFitness - previousBestFitness) < 0.001f)
            noChangeIter++;
        else
            noChangeIter = 0;
        previousBestFitness = bestFitness;

#ifdef DEBUG
        //log message
        cout << "#" << generationNumber<< " Fitness: " << bestFitness << \
        " Iterations without change: " << noChangeIter << endl;

        //log polynomial coefficients for visualizing   
        float *solution_tmp;
        if(generationNumber == 1)
            solution_tmp = new float[INDIVIDUAL_LEN];
        for(int i=0; i<INDIVIDUAL_LEN; i++){
            cudaMemcpy(&solution_tmp[i], &population_dev[i*POPULATION_SIZE],
                       sizeof(float), cudaMemcpyDeviceToHost);
            check_cuda_error("Coping solution to host");
        }

        cout << solution_tmp[0] << " " << solution_tmp[1] << " "
             << solution_tmp[2] << " " << solution_tmp[3] << endl;     
#endif

	}

    int t2 = clock(); //stop timer

    cout << "------------------------------------------------------------" << endl;    
    cout << "Finished! Found Solution:" << endl;

    //get solution from device to host
    float *solution = new float[INDIVIDUAL_LEN];
    for(int i=0; i<INDIVIDUAL_LEN; i++){
        cudaMemcpy(&solution[i], &population_dev[i*POPULATION_SIZE],
                   sizeof(float), cudaMemcpyDeviceToHost);
    }
    check_cuda_error("Coping solution to host");
    
    //solution is first individual of population with the best params of a polynomial    
    for(int i=0; i<INDIVIDUAL_LEN; i++){    
        cout << "\tc" << i << " = " << solution[i] << endl;
    }

    cout << "Best fitness: " << bestFitness << endl
		 << "Generations: " << generationNumber << endl;

    cout << "Time for GPU calculation equals \033[35m"
        << (t2-t1)/(double)CLOCKS_PER_SEC << " seconds\033[0m" << endl;

#ifdef PERF_METRIC
    //process performance numbers, mean and stddev
    double sum = 0;
    for (int i = 0; i < generationNumber; i++)
    {
        sum = sum + t_fitness[i];
    }
    double average = sum / generationNumber;

    sum = 0;
    for (int i = 0; i < generationNumber; i++)
    {
        sum = sum + pow((t_fitness[i] - average), 2);
    }
    double variance = sum / generationNumber;
    double std_deviation = sqrt(variance);

    cout << "Performance: " << average << " GPoints/s with stddev " << std_deviation << endl;
#endif

    delete [] points;
    delete [] solution;
#ifdef PERF_METRIC
    delete [] t_fitness; 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);       
#endif

    cudaFree(points_dev);//input points
    cudaFree(fitness_dev);//fitness array
    cudaFree(indexes_dev);//key for sorting
    cudaFree(population_dev);
    cudaFree(newPopulation_dev);
    cudaFree(state_random);//state curand
    cudaFree(mutIndivid_d);//mutation probability
    cudaFree(mutGene_d);//mutation probability

    curandDestroyGenerator(generator);
    
    return 0;
}

//------------------------------------------------------------------------------

static float *readData(const char *file_name, int *N_POINTS)
{
    std::ifstream is(file_name);
    if(!is.is_open())
    {
      cerr << "Error opening file " << file_name << endl;
      exit(1);
    }

    std::istream_iterator<double> start(is), end;
    std::vector<double> points(start, end);

    cout << "Reading file - success!" << endl;

    *N_POINTS = points.size()/2;

    float *points_arr = new float[points.size()];

    //rearrange points array so that first half contains x values, the other f(x)
    int i = 0;
    int N = points.size()/2;
    for (std::vector<double>::iterator it = points.begin() ; it != points.end(); ++it)
    {
        if (i % 2 == 0)
            points_arr[i/2] = *it;
        else
            points_arr[i/2 + N] = *it;

        i++;
    }

    return points_arr;
}

