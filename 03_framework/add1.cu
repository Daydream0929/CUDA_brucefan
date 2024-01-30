#include <math.h>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *hx = (double*)malloc(M);
    double *hy = (double*)malloc(M);
    double *hz = (double*)malloc(M);

    for (int i = 0; i < N; i ++) {
        hx[i] = a;
        hy[i] = b;
    }

    double *dx, *dy, *dz;
    cudaMalloc((void **)&dx, M);
    cudaMalloc((void **)&dy, M);
    cudaMalloc((void **)&dz, M);
    cudaMemcpy(dx, hx, M, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, M, cudaMemcpyHostToDevice);


    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(dx, dy, dz);

    cudaMemcpy(hz, dz, M, cudaMemcpyDeviceToHost);

    check(hz, N);

    free(hx);
    free(hy);
    free(hz);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    return 0;

}

__global__ void add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{   
    bool has_error = false;
    for (int i = 0; i < N; i ++) {
        if (fabs(z[i] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}