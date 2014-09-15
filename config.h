#define POPULATION_SIZE (4096*16) /* must be multiple of 64 == BLOCK */
#define INDIVIDUAL_LEN 4
#define N_POINTS 100

#define maxGenerationNumber 1500
#define maxConstIter 150
#define targetErr (N_POINTS*0.005)
#define mu_individuals 0.5
#define sigma_individuals 0.66
#define mu_genes 0.56
#define sigma_genes 0.75

// Emulate multi-process MPI on a single GPU, e.g. a laptop.
// Uncomment if-clause to disable.
// #if 0
#define cudaSetDevice(deviceID) cudaSetDevice(0)
// #endif

