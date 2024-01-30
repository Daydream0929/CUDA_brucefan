#include <math.h>
#include <cstdio>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add1(const double *x, const double *y, double *z, const int N);
void __global__ add2(const double *x, const double *y, double *z, const int N);
void __global__ add3(const double *x, const double *y, double *z, const int N);
void check(const double *z, int N);


// 版本1
__device__ double add1_device(const double x, const double y)
{
    return (x + y);
}

__global__ void add1(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = add1_device(x[n], y[n]);
    }
}

// 版本2
void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y;
}

__global__ void add2(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add2_device(x[n], y[n], &z[n]);
    }
}

// 版本3
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

__global__ void add3(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);
    }
}

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

    add1<<<grid_size, block_size>>>(dx, dy, dz, N);
    cudaMemcpy(hz, dz, M, cudaMemcpyDeviceToHost);
    check(hz, N);

    add2<<<grid_size, block_size>>>(dx, dy, dz, N);
    cudaMemcpy(hz, dz, M, cudaMemcpyDeviceToHost);
    check(hz, N);

    add3<<<grid_size, block_size>>>(dx, dy, dz, N);
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