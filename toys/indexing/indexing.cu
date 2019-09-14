#include <stdio.h>

// Not working currently.
__global__ void cuda_saxpy_mat_1(int N, float a, float *x, float *y)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	printf("k");
	if (i < N && j < N)
	{
		int index = i + j*N;
		y[index] = a* x[index] + y[index];
	}
}

__global__ void cuda_saxpy_mat_2(int N, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < N && j < N)
	{
		int index = i + j*N;
		y[index] = a* x[index] + y[index];
	}
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() 
{
	int N = 1<<10;

	printf("N = %d\n\n", N);


	float *host_x, *host_y, *device_x, *device_y;
	host_x = (float*)malloc(N*N*sizeof(float));
	host_y = (float*)malloc(N*N*sizeof(float));


	cudaMalloc(&device_x, N*N*sizeof(float)); 
	cudaMalloc(&device_y, N*N*sizeof(float));

	dim3 blocks_per_grid((N+255)/256, (N+255)/256, 1);
	dim3 threads_per_block(256, 256, 1);

	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < N; i++) 
		{
			*(host_x + i*N + j) = 1.0f;
			*(host_y + i*N + j) = 2.0f;
		}
	}

	printf("Initializing...\n");
	printf("x[0][0]     = %f\n", *(host_x + 0*N + 0));
	printf("...\n");
	printf("x[N-1][N-1] = %f\n\n", *(host_x + (N-1)*N + (N-1)));

	printf("y[0][0]     = %f\n", *(host_y + 0*N + 0));
	printf("...\n");
	printf("y[N-1][N-1] = %f\n\n", *(host_y + (N-1)*N + (N-1)));



	cudaMemcpy(device_x, host_x, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, host_y, N*N*sizeof(float), cudaMemcpyHostToDevice);

	printf("Computing y[i][j] = 6*x[i][j] + y[i][j]\n\n");

	cuda_saxpy_mat_1<<<blocks_per_grid,threads_per_block>>>(N, 6.0f, device_x, device_y);

	cudaMemcpy(host_y, device_y, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	printf("Outputting...\n");
	printf("x[0][0]     = %f\n", *(host_x + 0*N + 0));
	printf("...\n");
	printf("x[N-1][N-1] = %f\n\n", *(host_x + (N-1)*N + 0*(N-1)));

	printf("y[0][0]     = %f\n", *(host_y + 0*N + 0));
	printf("...\n");
	printf("y[N-1][N-1] = %f\n\n", *(host_y + (N-1)*N + (N-1)));

	cudaFree(device_x);
	cudaFree(device_y);

	free(host_x);
	free(host_y);

	printf("Look how fast that was!\n");

	return 0;
}
