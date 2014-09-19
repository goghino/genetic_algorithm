// Gets last error and prints message when error is present
static void check_cuda_error(const char *message);


////////////////////////////////////////////////////////////////////////////////
//                            GPU Kernels
////////////////////////////////////////////////////////////////////////////////

/**
    Initializes seed for CUDA random generator
*/
__global__ void initCurand(curandState *state)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(6482, idx, 0, &state[idx]);
}

/**
    Initializes initial population by random values from range <-5.0, 5.0>
*/
__global__ void initPopulation(float *population, curandState *state)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= POPULATION_SIZE) return;

    curandState localState = state[idx];

    for (int i = 0; i < INDIVIDUAL_LEN; i++)
        population[idx + i * POPULATION_SIZE] = 10 * curand_uniform(&localState) - 5;        

    state[idx] = localState;
}

/**
    An individual fitness function is the difference between measured f(x) and
    approximated polynomial g(x), built using individual's coeficients,
    evaluated on input data points.

    Smaller value means bigger fitness

    @size - number of individuals in current (sub)population
*/
__global__ void fitness_evaluate(float *individuals, float *points, float *fitness, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    float sumError = 0.0f;

    //for every given data point
	for (int pt = 0; pt < N_POINTS; pt++)
	{
		float f_approx = 0.0f;
		
        //for every polynomial parameter: Ci * x^(order)
		for (int order = 0; order < INDIVIDUAL_LEN; order++)
		{
			f_approx += individuals[idx + order * size] * pow(points[pt], order);
		}

		sumError += pow(f_approx - points[N_POINTS + pt], 2);
	}
	
    //The lower value of fitness is, the better individual fits the model
	fitness[idx] = sumError;
}

/**
    Creates new individuals by mating fittest individuals from currend population.
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
	int parent1_i = (curand(&localState) % (POPULATION_SIZE/2));
	int parent2_i = (curand(&localState) % (POPULATION_SIZE/2));


    //select crosspoint, do not select beginning and end of individual as crosspoint
	int crosspoint = curand(&localState) % (INDIVIDUAL_LEN - 2) + 1 ;
	state[idx] = localState;

    //do actual crossover
    for(int j=0; j<crosspoint; j++)
    {
            population_dev[idx +j*POPULATION_SIZE]
                = population_dev[parent1_i + j*POPULATION_SIZE];
    }

    for (int j = crosspoint; j < INDIVIDUAL_LEN; j++)
    {

        population_dev[idx + j*POPULATION_SIZE]
            = population_dev[parent2_i + j*POPULATION_SIZE];
    
    }
}

/**
    Mutation is addition of noise to genes, given mean and stddev.

    Probabilities of mutating individuals and their 
    genes is computed before calling this kernel
    @mutGene
    @mutIndivid

    @size - number of individuals in current (sub)population

    Mutation explained :
    individual == [1.0 1.0 1.0 1.0]
    mutNumber = 2
    loop 2 times:
       1st: num_of_bit_to_mutate = 2
            inverse individuals[2]   ->   [1.0 1.0 1.0+noise 1.0] 
       2nd: num_of_bit_to_mutate = 0
            inverse individuals[0]   ->   [1.0+noise 1.0 1.0+noise 1.0]
    return mutated individual is [1.0+noise 1.0 1.0+noise 1.0]               
*/
__global__ void mutation(float *individuals, curandState *state,
                         float* mutIndivid, float* mutGene, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    //first individual is not mutated to keep the best solution unchanged    
    if ((idx >= size) || (idx < 1))
        return;

    curandState localState = state[idx];

    float mutationRate = mutIndivid[idx];

    for (int j = 0; j < INDIVIDUAL_LEN; j++)
    {
        int flip_idx = idx + j * size;
        //probability of mutating gene 
        if (mutGene[flip_idx] < mutationRate)
        {
            individuals[flip_idx] += 0.01f * (2 * curand_uniform(&localState) - 1);
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
    if (idx >= POPULATION_SIZE) 
        return;

    indexes[idx] = idx;
}

/*
    Rearranges @population according to @indexes,
    indexes are keys created in key-value sorting array of fitness values

	individuals with small (good) fitness value are put to the beginning 
	individuals with large (bad) fitness value are placed at the end;
*/
__global__ void selection(float *population, float *newPopulation, int* indexes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //only first half needs to be placed in sorted manner
    //second half will be overwritten anyway
    if (idx > POPULATION_SIZE / 2)
        return;

    //reorder population so that fittest individuals are first
    for (int j = 0; j < INDIVIDUAL_LEN; j++)
    {
        newPopulation[idx + j * POPULATION_SIZE]
            = population[indexes[idx] + j * POPULATION_SIZE];
    }
}


////////////////////////////////////////////////////////////////////////////////
//                            Other GPU functions 
////////////////////////////////////////////////////////////////////////////////


/**
    Generates probabilities for mutation of individuals and their genes
    into arrays in device global memory using curand generator.
*/
static void generateMutProbab(float** mutIndivid, float **mutGene,
            curandGenerator_t generator, int size)
{
    //mutation rate of individuals
    curandGenerateNormal(generator, *mutIndivid,
                        size, mu_individuals, sigma_individuals);
    check_cuda_error("Error in normalGenerating 1");

    //mutation rate of each gene
    curandGenerateNormal(generator, *mutGene,
                        size * INDIVIDUAL_LEN, mu_genes, sigma_genes);
    check_cuda_error("Error in normalGenerating 2");
}
