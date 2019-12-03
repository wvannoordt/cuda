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

    for (int i = 0; i < grid_dimension*grid_dimension; i++)
    {
        if (i < 3 || i > grid_dimension*grid_dimension - 3)
        {
            std::cout << "Block " << i << " handles x_" << i*BLOCK_SIZE*BLOCK_SIZE << " thru x_" << i*BLOCK_SIZE*BLOCK_SIZE + BLOCK_SIZE*BLOCK_SIZE - 1 << std::endl;
        }
        else if (i == 4)
        {
            std::cout << "..." << std::endl;
        }
    }
    return 0;
}
