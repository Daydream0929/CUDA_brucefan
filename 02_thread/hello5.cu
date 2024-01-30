#include <iostream>

__global__ void hello_from_gpu()
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block- (%d %d) and thread- (%d %d) \n", bx, by, tx, ty);
}

int main()
{
    const dim3 grid_size(2, 3);
    const dim3 block_size(2, 4);
    hello_from_gpu<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}