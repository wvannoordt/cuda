//Thread block size
#define BLOCK_SIZE 16

//Matrices are stored in row-major order:
//M (row, col) = *(M.elements + row*M.stride + col)

//The stride field has been added so that sub-matrices can be efficiently represented.


#include "main_func.h"


//Note that functions prefized with __device__ are designated to run on the device...
//Get/Set matrix elements
__device__ specified_precision GetElement(const Matrix A, int row, int col)
{
	return A.elements[row*A.stride + col];
}
__device__ specified_precision SetElement(const Matrix A, int row, int col, specified_precision set_value)
{
	return A.elements[row*A.stride + col] = set_value;
}

//Returns the BLOCK_SIZE x BLOCK_SIZE matrix Asub of A that is
//located (col) sub-matrices to the right and (row) sub-matrices down from upper-left corner of A.
__device__ Matrix GetSubMatrix(const Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride*BLOCK_SIZE*row + BLOCK_SIZE*col];
	return Asub;
}

//Prototype
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	//Load A, B to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(specified_precision);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width;
	d_B.stride = B.stride;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(specified_precision);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	//Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.stride = C.stride;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(specified_precision);
	cudaMalloc(&d_C.elements, size);
	
	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width/dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	//Read C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	//Deallocate on device
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//Kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
	//Block row and column (see notes / Cuda programming guide)
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	
	//Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	
	//Each thread computes one welement of Csub by accumulating results into thread_c_value
	float thread_c_value = 0;
	
	//thread row and column index within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	//Loop over all sub-matrices of A and B required to compute Csub, multiply
	//each pair of sub-matrices together and accumulate results
	for (int m = 0; m < (A.width / BLOCK_SIZE); m++)
	{
		//Get sub-matrices. Rember that these matrices are actually very lightweight objects 
		//since they don't actually contain their own entries, just the address of the first one.
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		
		//Shared memory used to store Asub and Bsub respectively
		__shared__ specified_precision As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ specified_precision Bs[BLOCK_SIZE][BLOCK_SIZE];
		
		//Load Asub and Bsub from device memory to chared memory.
		//Each thread loads one element of each sub-matrix.
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		
		//Sync threads to make sure sub-matrices are complete before computing.
		__syncthreads();
		
		//Multiply Asub and Bsub together, each thread computes one entry
		for (int idx = 0; idx < BLOCK_SIZE; idx++)
		{
			thread_c_value += As[row][idx]*Bs[idx][col];
		}
		
		//Synchronize to ensure that preceding computation is done before loading two new sub-matrices in the 
		//next iteration.
		__syncthreads();
	}
	//Write Csub to decive memory
	//each thread writes one element
	SetElement(Csub, row, col, thread_c_value);
}














