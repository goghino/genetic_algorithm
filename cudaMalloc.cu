#include <stdio.h>
#include "check.h"

#if defined(__CUDACC__) && defined(THRUST_REUSE_MALLOC)

// Thrust kernels allocate some GPU memory for internal use.
// This file implements cudaMalloc/cudaFree hooks to re-use a single
// GPU memory allocation for all Thrust kernel calls, eliminating
// subsequent allocations-deallocations.
__thread bool cudaMallocReuse = false;

// Nnumber of supported mallocs existing at the same time to be reused
const int N = 4;
static __thread int idx = 0;


static __thread void* cudaMallocPtr[N];
static __thread size_t cudaMallocSize[N];
static __thread bool cudaMallocInitialized[N] = {0};

// Functions that actually frees our reused malloc`ed memory
// Works only for N == 4
static void cudaFreeReused1();
static void cudaFreeReused2();
static void cudaFreeReused3();
static void cudaFreeReused4();
void (*cudaFreeReused[N])() =
            {cudaFreeReused1, cudaFreeReused2, cudaFreeReused3, cudaFreeReused4};


////////////////////////////////////////////////////////////////////////////////

// undefined reference to __real_cudaMalloc will be resolved to cudaMalloc by linker
extern "C" cudaError_t __real_cudaMalloc(void **devPtr, size_t size);

// undefined reference to cudaMalloc will be resolved to __wrap_cudaMalloc
extern "C" cudaError_t __wrap_cudaMalloc(void **devPtr, size_t size)
{
	// Call real cudaMalloc, if we are not in reuse mode.
	if (!cudaMallocReuse)
	{
        __real_cudaMalloc(devPtr, size);
		check_cuda_error("Calling real cudaMalloc\n");
		return cudaSuccess;
	}
	
    // Here we are in the reuse mode
	if (idx >= N || idx < 0)
	{
		fprintf(stderr, "Trying to use existing cudaMalloc more times than allowed!\n");
		exit(1);
	}
	
    // Check if we are calling reusable cudaMalloc for first time to actually call real cudaMalloc
	if (!cudaMallocInitialized[idx])
	{
        __real_cudaMalloc(&cudaMallocPtr[idx], size);
		check_cuda_error("Calling real cudaMalloc in test if initialized\n");
		cudaMallocSize[idx] = size;
		atexit(cudaFreeReused[idx]);
		cudaMallocInitialized[idx] = true;
	}
	
	if (size != cudaMallocSize[idx])
	{
		fprintf(stderr, "Exisiting cudaMalloc size (%zu) does not match the requested size (%zu)!\n",
			cudaMallocSize[idx], size);
		exit(1);
	}
		
	*devPtr = cudaMallocPtr[idx];
    idx++;
	
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" cudaError_t __real_cudaFree(void* devPtr);

extern "C" cudaError_t __wrap_cudaFree(void* devPtr)
{
	// Call real cudaFree, if we are not in reuse mode.
	if (!cudaMallocReuse)
	{
		// Ignore cudaErrorCudartUnloading error here.
		// Due to, perhaps, dlopen of library containing CUDA kernels, which
		// originates from runtime optimization, the CUDA-destructors order
		// becomes broken. No idea what to do with this, so we simply let CUDA
		// runtime to shoot its own leg, if it wants to do so, no problem!
		cudaError_t status = __real_cudaFree(devPtr);
		if (status != cudaErrorCudartUnloading)
            check_cuda_error("cudaErrorCudartUnloaging\n");
		return cudaSuccess;
	}

    // After last allowed malloc, idx == N, we need to work with idx-1
	if (idx > N || idx <= 0)
	{
		fprintf(stderr, "Trying to release existing cudaMalloc more times than allowed\n");
		exit(1);
	}

	if (!cudaMallocInitialized[idx-1])
	{
		fprintf(stderr, "Cannot cudaFree uninitialized cudaMalloc!\n");
		exit(1);
	}
	
    //simulate cudaFree by lowering the index to pre-allocated memory chunks array
    idx--;
	
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////

void cudaFreeReused1()
{
    __real_cudaFree(cudaMallocPtr[0]);
	check_cuda_error("Real cudaFree at cudaFreeReused1()");
}

void cudaFreeReused2()
{
    __real_cudaFree(cudaMallocPtr[1]);
	check_cuda_error("Real cudaFree at cudaFreeReused2()");
}

void cudaFreeReused3()
{
    __real_cudaFree(cudaMallocPtr[2]);
	check_cuda_error("Real cudaFree at cudaFreeReused3()");
}

void cudaFreeReused4()
{
    __real_cudaFree(cudaMallocPtr[3]);
	check_cuda_error("Real cudaFree at cudaFreeReused4()");
}

#endif

