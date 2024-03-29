#include <math.h>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*)malloc(M);
    double *y = (double*)malloc(M);
    double *z = (double*)malloc(M);

    for (int i = 0; i < N; i ++) {
        x[i] = a;
        y[i] = b;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);
    free(y);
    free(z);

    return 0;

}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int i = 0; i < N; i ++) {
        z[i] = x[i] + y[i];
    }
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