void check_cuda_error(const char *message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
		exit(1);
	}
}
