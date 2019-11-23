#include "stdio.h"
#include <iostream>


#ifndef PRECISION
#define PRECISION 1
#endif

#if(PRECISION==1)
typedef float specified_precision;
#else
typedef double specified_precision;
#endif

typedef struct
{
	int width;
	int height;
	int stride;
	specified_precision* elements;
} Matrix;

void MatMul(const Matrix, const Matrix, Matrix);

void H_set_element(Matrix M, int row, int col, specified_precision elem)
{
	*(M.elements + row*M.stride + col) = elem;
}

specified_precision H_get_element(Matrix M, int row, int col)
{
	return *(M.elements + row*M.stride + col);
}

Matrix init_rand_matrix_blocksize(int row_blocks, int col_blocks)
{
	Matrix A;
	specified_precision* a_base;
	
	A.width = col_blocks*BLOCK_SIZE;
	A.height = row_blocks*BLOCK_SIZE;
	A.stride = col_blocks*BLOCK_SIZE;
	
	size_t a_elem_size = A.width*A.height*sizeof(specified_precision);
	a_base = (specified_precision*)malloc(a_elem_size);
	
	A.elements = a_base;
	
	for (int i = 0; i < A.height; i++)
	{
		for (int j = 0; j < A.width; j++)
		{
			specified_precision entry = (specified_precision)rand()/RAND_MAX;
			H_set_element(A, i, j, entry);
		}
	}
	
	return A;
}

Matrix init_empty_matrix_blocksize(int row_blocks, int col_blocks)
{
	Matrix A;
	specified_precision* a_base;
	
	A.width = col_blocks*BLOCK_SIZE;
	A.height = row_blocks*BLOCK_SIZE;
	A.stride = col_blocks*BLOCK_SIZE;
	
	size_t a_elem_size = A.width*A.height*sizeof(specified_precision);
	a_base = (specified_precision*)malloc(a_elem_size);
	
	A.elements = a_base;
	return A;
}

int main()
{

	int N_blocks = 500;
	
	Matrix A = init_rand_matrix_blocksize(N_blocks, N_blocks);
	Matrix B = init_rand_matrix_blocksize(N_blocks, N_blocks);
	Matrix C = init_empty_matrix_blocksize(N_blocks, N_blocks);
	
	MatMul(A,B,C);
	
	std::cout << H_get_element(A, 0, 0) << std::endl;
	std::cout << H_get_element(B, 0, 0) << std::endl;
	std::cout << H_get_element(C, 1, 0) << std::endl;
	
	free(A.elements);
	free(B.elements);
	free(C.elements);
	
	
	return 0;
}


