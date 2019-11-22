//Kernel definition (notice the use of single-precision!)
__global__ void MadAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < N && j < N)
	{
		C[i][j] = A[i][j] + B[i][j];
	}
}

int main()
{
	//stuff
	
	//Kernel invocation
	//each block has 16x16 threads
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y)
	MatAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);
}
