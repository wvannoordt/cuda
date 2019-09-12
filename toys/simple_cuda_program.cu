#include <stdio.h>

__global__ void cuda_test(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i] + i;
}

int main() 
{
	// initialize N = 2^20 (<< is the shift operator!)
	int N = 1<<20;
	
	// output
	printf("N = %d\n\n", N);
	
	
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float)); 
	cudaMalloc(&d_y, N*sizeof(float));
	
		
	dim3 blocks_per_grid((N+255)/256, 1, 1);
	dim3 threads_per_block(256, 1, 1);
	
	for (int i = 0; i < N; i++) 
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	
	printf("Initializing...\n");
	printf("x[0]     = %f\n", x[0]);
	printf("...\n");
	printf("x[N - 1] = %f\n\n", x[N - 1]);
	
	printf("y[0]     = %f\n", y[0]);
	printf("...\n");
	printf("y[N - 1] = %f\n\n", y[N - 1]);

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
	
	printf("Computing y[i] = 2*x[i] + y[i] + i\n\n");
	
    cuda_test<<<blocks_per_grid,threads_per_block>>>(N, 2.0f, d_x, d_y); 
    
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Outputting...\n");
	printf("x[0]     = %f\n", x[0]);
	printf("...\n");
	printf("x[N - 1] = %f\n\n", x[N - 1]);
	
	printf("y[0]     = %f\n", y[0]);
	printf("...\n");
	printf("y[N - 1] = %f\n", y[N - 1]);
    
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

    return 1;
}
