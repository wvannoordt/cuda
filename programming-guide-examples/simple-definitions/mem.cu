//device code
__global__ void VecAdd(float* a, float* b, float* c, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	{
		c[i] = a[i] + b[i];
	}
}

//host code
int main()
{
	int N = 1024;
	size_t size = N*sizeof(float);
	
	//allocate input vectors in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
	
	//initialize input vectors
	for (int i = 0; i < N; i++)
	{
		h_A[i] =  0;
		h_B[i] =  0;
	}
	
	//allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	
	//copy from host to device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	//Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N+threadsPerBlock - 1)/threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	
	//Copy result from device to host. h_C is the result.
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	//free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	//free host memory
	free(h_A);
	free(h_B);
	free(h_C);
}
