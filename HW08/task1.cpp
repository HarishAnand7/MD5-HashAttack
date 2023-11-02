#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "matmul.h"
#include <random>
#include <chrono>
#include <iostream>

using namespace std;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]); // Matrix size
    int t = atoi(argv[2]); // Number of threads

    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    const float min = -1.0, max = 1.0; // The range for the random number generator is -1.0 to 1.0
    uniform_real_distribution<float> dist(min, max);

    float *A = (float*)malloc(n * n * sizeof(float));
    float *B = (float*)malloc(n * n * sizeof(float));
    float *C = (float*)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++)
    {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;

    omp_set_num_threads(t);
    start = high_resolution_clock::now();

    mmul(A, B, C, n); // Perform matrix multiplication
    end = high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    printf("%f\n%f\n%f\n", C[0], C[n*n-1], duration.count() * 1000.0);
    free(A);
    free(B);
    free(C);

    return 0;
}
 
