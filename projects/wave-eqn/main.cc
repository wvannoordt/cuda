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

#define BLOCK_SIZE 32


void single_to_double_idx(int i, int* x, int* y, int domain_dim)
{
    *(x) = i%domain_dim;
    *(y) = i/domain_dim;
}
void double_to_single_idx(int x, int y, int* idx, int domain_dim)
{
    *(idx) = y*domain_dim + x;
}


int main()
{
    int domain_width_points = 1 << 9;
    int grid_dimension = domain_width_points / BLOCK_SIZE;

    std::cout << "Domain:                         " << domain_width_points << "x" << domain_width_points << std::endl;
    std::cout << "Block size:                     " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
    std::cout << "Solution vector size:           " << domain_width_points*domain_width_points << std::endl;
    std::cout << "Solution vector entries/block:  " << BLOCK_SIZE*BLOCK_SIZE << std::endl;
    std::cout << "Total block count:              " << grid_dimension*grid_dimension << std::endl;
    std::cout << "Grid size:                      " << grid_dimension << "x" << grid_dimension << std::endl << std::endl;

    specified_precision domain_length_x = 1.0;
    specified_precision domain_length_y = 1.0;
    specified_precision inverse_dx = domain_width_points/domain_length_x;
    specified_precision inverse_dy = domain_width_points/domain_length_y;

    std::cout << "E.g. entry (0,0) is a function of entry (1,0) and (0,1)," << std::endl;
    int a, b, c;
    double_to_single_idx(0,0,&a,domain_width_points);
    double_to_single_idx(1,0,&b,domain_width_points);
    double_to_single_idx(0,1,&c,domain_width_points);
    std::cout << "so vector entry "<< a << " is a function of vector entry ";
    std::cout << b << " and vector entry " << c << "." << std::endl;
    
	int last_block_entry = BLOCK_SIZE*BLOCK_SIZE - 1;
	int x, y;
	std::cout << last_block_entry << std::endl;
	single_to_double_idx(last_block_entry, &x, &y, domain_width_points);
	std::cout << x << ", " << y << std::endl;

    return 0;
}
