//specifies function definition space
#define POPULATION_SIZE (4096*16*3) /* must be multiple of 64 == BLOCK */

//number of polynomial coefficients, including c0 coeff. (IL = max_order + 1)
#define INDIVIDUAL_LEN 4

//number of noisy data points to approximate
#define N_POINTS 100

/**        GA algorithm specific params         */

//maximum of numbers of generations
#define maxGenerationNumber 1500

//max. number of iterations without improvement of solution (stuck in local minima)
#define maxConstIter 150

//defines target error of GA, if best fitness is lower, GA terminates
#define targetErr (N_POINTS*0.005)

//probabilities of mutation of individual and its genes
#define mu_individuals 0.5
#define sigma_individuals 0.66
#define mu_genes 0.56
#define sigma_genes 0.75

//defines interval for random initialization of polynomial coefficients (-RndRange,+RndRange), RndRange > 0
#define RndRange 500


// Emulate multi-process MPI on a single GPU, e.g. a laptop.
// Uncomment if-clause to disable.
#if 0
#define cudaSetDevice(deviceID) cudaSetDevice(0)
#endif
