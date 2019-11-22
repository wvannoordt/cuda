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

void H_set_element(Matrix M, int row, int col, specified_precision elem)
{
	*(M.elements + row*M.stride + col) = elem;
}

specified_precision H_get_element(Matrix M, int row, int col)
{
	return *(M.elements + row*M.stride + col);
}

int main()
{
	Matrix A;
	int N_blocks = 5;
	specified_precision* a_base;
	
	A.width = N_blocks*BLOCK_SIZE;
	A.height = N_blocks*BLOCK_SIZE;
	A.stride = N_blocks*BLOCK_SIZE;
	
	size_t a_elem_size = A.width*A.height*sizeof(specified_precision);
	a_base = (specified_precision*)malloc(a_elem_size);
	
	A.elements = a_base;
	
	for (int i = 0; i < A.height; i++)
	{
		for (int j = 0; j < A.width; j++)
		{
			specified_precision entry = (specified_precision)rand()/RAND_MAX;
			H_set_element(A, i, j, entry);
			std::cout << entry << std::endl;
		}
	}
	
	free(a_base);
	
	return 0;
}


