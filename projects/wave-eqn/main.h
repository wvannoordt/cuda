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
    int domain_width = 1 << 9;
    int grid_dimension = domain_width / BLOCK_SIZE;
    std::cout << "Domain:                 " << domain_width << "x" << domain_width << std::endl;
    std::cout << "Block size:             " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
    std::cout << "Solution vector size:   " << domain_width*domain_width << std::endl;
    return 0;
}
