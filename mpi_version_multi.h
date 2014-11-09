#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "check.h"

// Forward declarations
extern "C" {
    // Reads input file with noisy points. Points will be approximated by 
    // polynomial function using GA.
    float *readData(const char *name, const int POINTS_CNT);

    // Finished MPI and aborts
    void my_abort(int err);

    void doInitPopulation(float *population_dev, curandState *state_random);

    void doInitCurandOnly(curandState *state_random, int size);

    void doCrossover(float *population_dev, curandState* state_random);

    void doMutation(float *population_dev, curandState *state_random, int size,
                float *mutIndivid_d, float *mutGene_d, curandGenerator_t generator);

    void doFitness_evaluate(float *population_dev, float *points_dev, int N_POINTS,
                            float *fitness_dev, int size);

    void doSelection(thrust::device_ptr<float>fitnesses_thrust,
                     thrust::device_ptr<int>indexes_thrust, int *indexes_dev,
                     float *population_dev, float* newPopulation_dev);

    void doTranspose(float *population_devT, float *population_dev, int size);

    void doTranspose_inverse(float *population_devT, float *population_dev, int size);

}
