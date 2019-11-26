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

#define BLOCK_SIZE 16

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


    return 0;
}
