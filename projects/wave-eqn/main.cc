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

    int x1,y1,x2,y2,init,term,x_inq,y_inq,l_bound,u_bound,depend;
    for (int i = 0; i < grid_dimension*grid_dimension; i++)
    {
        l_bound = grid_dimension + 1;
        u_bound = -1;
        init = i*BLOCK_SIZE*BLOCK_SIZE;
        term = i*BLOCK_SIZE*BLOCK_SIZE + BLOCK_SIZE*BLOCK_SIZE - 1;
        single_to_double_idx(init, &x1, &y1, domain_width_points);
        single_to_double_idx(term, &x2, &y2, domain_width_points);
        if (i < 3 || i > grid_dimension*grid_dimension - 4)
        {
            std::cout << "Block " << i << " handles y_" << init;
            std::cout << " thru y_" << term;
            std::cout << ", grid_idx_1=(" << x1 << "," << y1 << ")";
            std::cout << ", grid_idx_2=(" << x2 << "," << y2 << ")" << std::endl;

            for (int k = init; k <= term; k++)
            {
                single_to_double_idx(k,&x_inq,&y_inq,domain_width_points);
                if (x_inq+1 < domain_width_points)
                {
                    double_to_single_idx(x_inq+1,y_inq,&depend,domain_width_points);
                    l_bound = depend < l_bound ? depend: l_bound;
                    u_bound = depend > u_bound ? depend: u_bound;
                }
                if (y_inq+1 < domain_width_points)
                {
                    double_to_single_idx(x_inq,y_inq+1,&depend,domain_width_points);
                    l_bound = depend < l_bound ? depend: l_bound;
                    u_bound = depend > u_bound ? depend: u_bound;
                }
                if (x_inq-1 >= 0)
                {
                    double_to_single_idx(x_inq-1,y_inq,&depend,domain_width_points);
                    l_bound = depend < l_bound ? depend: l_bound;
                    u_bound = depend > u_bound ? depend: u_bound;
                }
                if (y_inq-1 >= 0)
                {
                    double_to_single_idx(x_inq,y_inq-1,&depend,domain_width_points);
                    l_bound = depend < l_bound ? depend: l_bound;
                    u_bound = depend > u_bound ? depend: u_bound;
                }
            }

            std::cout << "Requires x_" << l_bound << " thru x_" << u_bound << std::endl;
        }
        else if (i == 4)
        {
            std::cout << "..." << std::endl;
        }
    }
    return 0;
}
