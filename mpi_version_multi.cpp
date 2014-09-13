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

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <algorithm>

#include "mpi_version_multi.h"

using namespace std;

// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        cerr << "MPI error calling \""#call"\"\n"; \
        my_abort(-1); }

/**
    Generates probabilities for mutation of individuals and their genes
    into arrays in device global memory
*/
void generateMutProbab(float** mutIndivid, float **mutGene,
                       curandGenerator_t generator, int size);

/*
    ---------------------------------------------------------
    |  MPI communication and encapsulated GPU computation   |
    ---------------------------------------------------------
*/
int main(int argc, char **argv)
{
    if(argc != 2) {
        cerr << "Usage: $mpirun -np N ./gpu inputFile" << endl;    
        return -1;
    }

    // Initialize MPI state
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Get our MPI node number and node count
    int commSize, commRank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));
    int deviceID = commRank;

    if(commSize > 4)
    {
        cerr << "Cannot run with more than 4 processes!" << endl;
        return -1;        
    }

    //read input data
    //points are the data to approximate by a polynomial
    float *points = readData(argv[1], N_POINTS);
    if(points == NULL)
        return -1;

    /**
        Allocations of GPU memory
        
        Some portion of the code (crossover, selection)
        is executed only on master process, some workload
        (mutation, fitness evaluation) is distributed amongst
        all existing processes.

        -master process allocates data for whole population
        -slave processes only for its own local data part
    */

    //set proper device
    cudaSetDevice(deviceID);
    check_cuda_error("Setting device");

    //device memory for holding input points on all processes
    float *points_dev;
    cudaMalloc(&points_dev, 2*N_POINTS*sizeof(float)); // [x, f(x)+err]
    check_cuda_error("Error allocating device memory");
    cudaMemcpy(points_dev, points, 2*N_POINTS*sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error("Error copying data");


    //size of local portion of data
    //TODO population%commSize  != 0
    int size = POPULATION_SIZE/commSize;


    //arrays to hold population    
    float *population_dev;
    float *newPopulation_dev;
    if(commRank == 0){
        cudaMalloc(&population_dev, POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(float));
        check_cuda_error("Error allocating device memory");

        cudaMalloc(&newPopulation_dev, POPULATION_SIZE * INDIVIDUAL_LEN * sizeof(float));
        check_cuda_error("Error allocating device memory");
    }else{
        //TODO
        cudaMalloc(&population_dev, size * INDIVIDUAL_LEN * sizeof(float));
        check_cuda_error("Error allocating device memory");    
    }

    //arrays that keeps fitness of individuals withing current population
    float *fitness_dev;    
    if(commRank == 0){
        cudaMalloc(&fitness_dev, POPULATION_SIZE*sizeof(float));
        check_cuda_error("Error allocating device memory");
    }else{
        //TODO
        cudaMalloc(&fitness_dev, size * sizeof(float));
        check_cuda_error("Error allocating device memory");   
    }

    //curand random states
    curandState *state_random;
    if(commRank == 0){
        cudaMalloc((void **)&state_random,POPULATION_SIZE * sizeof(curandState));
        check_cuda_error("Allocating memory for curandState");
    }else{
        //TODO
        cudaMalloc( (void **)&state_random,
                    size * sizeof(curandState));
        check_cuda_error("Allocating memory for curandState");
    
    }

    //mutation probabilities, mutation done only locally
    //TODO
    float* mutIndivid_d;
    cudaMalloc((void **) &mutIndivid_d, size * sizeof(float));
    check_cuda_error("Allocating memory in mutIndivid_d");

    float* mutGene_d;
    //TODO
    cudaMalloc((void **)&mutGene_d, size * INDIVIDUAL_LEN*sizeof(float));
    check_cuda_error("Allocating memory in mutGene_d");

    //create PRNG for generating mutation probabilities
    curandGenerator_t generator;

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    check_cuda_error("Error in curandCreateGenerator");

    curandSetPseudoRandomGeneratorSeed(generator, time(NULL)/(commRank+1));
    check_cuda_error("Error in curandSeed");

    //key value for sorting, sorting done only on master process
    int *indexes_dev;
    thrust::device_ptr<int> indexes_thrust;
    thrust::device_ptr<float> fitnesses_thrust;
    if(commRank == 0){
        cudaMalloc(&indexes_dev, POPULATION_SIZE*sizeof(int));
        check_cuda_error("Error allocating device memory");

        //recast device pointers into thrust copatible pointers
        indexes_thrust = thrust::device_pointer_cast(indexes_dev);
        fitnesses_thrust = thrust::device_pointer_cast(fitness_dev);
    }



    /**
        Main GA loop
    */

    //Initialize first population (with zeros or some random values)
    if(commRank == 0)
        doInitPopulation(population_dev, state_random);
    
    int t1 = clock(); //start timer

    int generationNumber = 0;
    int noChangeIter = 0;

    float bestFitness = INFINITY;
    float previousBestFitness = INFINITY;

	while ( (generationNumber < maxGenerationNumber)
            /*&& (bestFitness > targetErr)
            && (noChangeIter < maxConstIter)*/ )
	{
		generationNumber++;
	
        /** crossover first half of the population and create new population */
        if(commRank == 0)
    		doCrossover(population_dev, state_random);

        /** distribute population to all processes to perform mutation and fitness eval*/
        //TODO
        MPI_CHECK(
            MPI_Scatter(population_dev, POPULATION_SIZE/commSize * INDIVIDUAL_LEN, MPI_FLOAT,
                        population_dev, POPULATION_SIZE/commSize * INDIVIDUAL_LEN, MPI_FLOAT,
                        0, MPI_COMM_WORLD)
        );

		/** mutate population and childrens in the local portion of population*/
        generateMutProbab(&mutIndivid_d, &mutGene_d, generator, size);
		doMutation(population_dev, state_random, mutIndivid_d, mutGene_d, size);

        /** evaluate fitness of individuals in local portion of population */
		doFitness_evaluate(population_dev, points_dev, fitness_dev, size);

        /** gather population & fitnesses back to master process to perform selection*/
        MPI_CHECK(
            MPI_Gather(population_dev, POPULATION_SIZE/commSize * INDIVIDUAL_LEN, MPI_FLOAT,
                       population_dev, POPULATION_SIZE/commSize * INDIVIDUAL_LEN, MPI_FLOAT,
                       0, MPI_COMM_WORLD)
        );

        MPI_CHECK(
            MPI_Gather(fitness_dev, POPULATION_SIZE/commSize, MPI_FLOAT,
                       fitness_dev, POPULATION_SIZE/commSize, MPI_FLOAT,
                       0, MPI_COMM_WORLD)
        );

        
        /** select individuals for mating to create the next generation,
            i.e. sort population according to its fitness and keep
            fittest individuals first in population  */
        if(commRank == 0) {
            doSelection(fitnesses_thrust, indexes_thrust, indexes_dev,
                        population_dev, newPopulation_dev);    

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
            cout << "Best fitness: " << bestFitness << endl;

            //log message
            #if defined(DEBUG)
            cout << "#" << generationNumber<< " Fitness: " << bestFitness << \
            " Iterations without change: " << noChangeIter << endl;
            #endif
        }

	}

    int t2 = clock(); //stop timer


    /**
       Process results on master process.
    */
    int masterProcess = 0;

    if(commRank == masterProcess)
    {
  
        cout << "------------------------------------------------------------" << endl;    
        cout << "Finished! Found Solution: " << endl;       

        //solution with the best params of a polynomial 
        for(int i=0; i<INDIVIDUAL_LEN; i++){ 
            float solution;            
            
            //get best individual
            cudaMemcpy(&solution, &population_dev[i*POPULATION_SIZE],
                       sizeof(float), cudaMemcpyDeviceToHost);
            check_cuda_error("Coping fitnesses_dev[0] to host");
        
            //print
            cout << "\tc" << i << " = " << solution << endl;
        }

        cout << "Best fitness: " << bestFitness << endl \
        << "Generations: " << generationNumber << endl;

        cout << "Time for GPU calculation equals \033[35m" \
            << (t2-t1)/(double)CLOCKS_PER_SEC << " seconds\033[0m" << endl;

    }

    
    /**
        Free memory
    */
    cudaFree(points_dev);//input points
    check_cuda_error("points free");
    
    cudaFree(fitness_dev);//fitness array
    check_cuda_error("fitness free");

    if(commRank == 0){
        cudaFree(indexes_dev);//key for sorting
        check_cuda_error("indexes free");
    }

    cudaFree(population_dev);
    check_cuda_error("population free");

    if(commRank == 0){
        cudaFree(newPopulation_dev);
        check_cuda_error("newPopulation free");
    }
    cudaFree(state_random);//state curand
    check_cuda_error("state free");

    cudaFree(mutIndivid_d);//mutation probability
    check_cuda_error("mutInd free");

    cudaFree(mutGene_d);//mutation probability
    check_cuda_error("mutGene free");

    curandDestroyGenerator(generator);
    check_cuda_error("Destroying generator");

    cudaDeviceReset();
    check_cuda_error("Resseting device");


    MPI_CHECK(MPI_Finalize());
}

//------------------------------------------------------------------------------

float *readData(const char *name, const int POINTS_CNT)
{
    FILE *file = fopen(name,"r");
 
	float *points = new float[2*POINTS_CNT]; 
    if (file != NULL){

        int k=0;
        //x, f(x)
        while(fscanf(file,"%f %f",&points[k],&points[POINTS_CNT+k])!= EOF){
            k++;
        }
        fclose(file);
        cout << "Reading file - success!" << endl;
    }else{
        cerr << "Error while opening the file " << name << "!!!" << endl;
        delete [] points;
        return NULL;
    }

    return points;
}

/**
    Generates probabilities for mutation of individuals and their genes
    into arrays in device global memory
*/
void generateMutProbab(float** mutIndivid, float **mutGene, curandGenerator_t generator, int size)
{
    //mutation rate of individuals
    curandGenerateNormal(generator, *mutIndivid,
                        size, mu_individuals, sigma_individuals);
    check_cuda_error("Error in normalGenerating 1");

    //mutation rate of each gene
    curandGenerateNormal(generator, *mutGene,
                        size*INDIVIDUAL_LEN, mu_genes, sigma_genes);
    check_cuda_error("Error in normalGenerating 2");
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}
