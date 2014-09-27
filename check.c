#include "check.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

void check_cuda_error(const char *message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
		exit(1);
	}
}


//NOT WORKING, undefined references during linking stage!!!!
//files gpu_version, mpi_version_multi.cu has definition of this function hardcoded!!!
