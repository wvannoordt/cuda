#include <stdio.h>

// This is the interesting part. The specifier "__global__" indicates that this function runs in the GPU
// kernel and thus runs in a different environment than the main function. Remember, C functions need
// to be prototyped! 
__global__ void cuda_test(int n, float a, float *x, float *y)
{
	// This line is very important. Not doing this properly can greatly affect the performence of the
	// program (see"memory coalescing")

	// Note also that blockIdx, blockDim, and ThreadIdx are all CUDA built-ins. They are all of the "dim3"
	// type, which is also a CUDA built-in type.
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i] + i;
}

int main() 
{
	// Initialize N = 2^20 (<< is the shift operator!)
	int N = 1<<20;

	// Output
	printf("N = %d\n\n", N);


	// Create two pars of arrays. One copy is for the host (CPU), the other copy is for the device (GPU).
	float *host_x, *host_y, *device_x, *device_y;

	// Use dynamic allocation here since the heap memory is larger than the stack memory.
	host_x = (float*)malloc(N*sizeof(float));
	host_y = (float*)malloc(N*sizeof(float));

	// Allocate the device memory
	cudaMalloc(&device_x, N*sizeof(float)); 
	cudaMalloc(&device_y, N*sizeof(float));

	// Give the parallel structure of the problem. The triplet gives block count in each parallelization direction.
	dim3 blocks_per_grid((N+255)/256, 1, 1);
	dim3 threads_per_block(256, 1, 1);

	// Initialize some example data.
	for (int i = 0; i < N; i++) 
	{
		host_x[i] = 1.0f;
		host_y[i] = 2.0f;
	}

	// Output.
	printf("Initializing...\n");
	printf("x[0]     = %f\n", host_x[0]);
	printf("...\n");
	printf("x[N - 1] = %f\n\n", host_x[N - 1]);

	printf("y[0]     = %f\n", host_y[0]);
	printf("...\n");
	printf("y[N - 1] = %f\n\n", host_y[N - 1]);



	// Copy the initialized example data from host to the device. Note that the DESTINATION is always the first argument and
	// the SOURCE is always the second argument.

	// cudaMemcpyHostToDevice is a CUDA built-in that specifies transfer direction.
	cudaMemcpy(device_x, host_x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, host_y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Output.
	printf("Computing y[i] = 2*x[i] + y[i] + i\n\n");

	// Call the function on the GPU. At this point it should be noted that the variables "device_x" and "device_y"
	// do not exist on the CPU, and so an access to this will cause a segmentation fault!

	// This is the line everyone is here for.
	cuda_test<<<blocks_per_grid,threads_per_block>>>(N, 2.0f, device_x, device_y); 

	// Copy the computed data back to the host as "host_y".
	cudaMemcpy(host_y, device_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	// Output.
	printf("Outputting...\n");
	printf("x[0]     = %f\n", host_x[0]);
	printf("...\n");
	printf("x[N - 1] = %f\n\n", host_x[N - 1]);

	printf("y[0]     = %f\n", host_y[0]);
	printf("...\n");
	printf("y[N - 1] = %f\n", host_y[N - 1]);

	// Free the memory. "cudaFree" is a CUDA built-in.
	cudaFree(device_x);
	cudaFree(device_y);

	// Free the memory.
	free(host_x);
	free(host_y);

	// Haha
	printf("Look how fast that was!\n");

	// Exit code.
	return 0;
}
