//Matrices are stored in row-major order:
//M (row, col) = *(M.elements + row*M.width + col)

#define BLOCK_SIZE 16

#define KERNEL_DEBUG 0

#include "main_func.h"

//Forward declaration of kernel! (prototyping)
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	//Load A, B to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(specified_precision);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(specified_precision);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	//Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(specified_precision);
	cudaMalloc(&d_C.elements, size);

	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width/dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);


#if (KERNEL_DEBUG ==1)
	cudaError_t cudaError;
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
	{
		printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
#endif
	//Read C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	//Deallocate on device
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//Kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
#if (KERNEL_DEBUG ==1)
	printf("Hello, world from the device!\n");
#endif
	//One element of C computed by one thread in the kernel.
	specified_precision thread_c_value = 0;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	for (int row_col_idx = 0; row_col_idx < A.width; row_col_idx++)
	{
		thread_c_value += A.elements[row*A.width + row_col_idx]*B.elements[row_col_idx*B.width + col];
	}
	C.elements[row*C.width+col] = thread_c_value;
}

//Notes:

//Each thread reads one row of A and one column of B, and in this implementation, A is read from global memory (B.width) times,
//while B is read from global memory (A.height) times. (This isn't actually entirely clear...).
