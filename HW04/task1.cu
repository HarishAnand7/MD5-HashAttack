#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include "matmul.cuh"


int main(int argc, char *argv[])
{
    int  n = std::atoi(argv[1]);
    unsigned int threads_per_block= std::atoi(argv[2]);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    float *a, *b, *c;
    a=(float*)malloc(n*n*sizeof(float));
    b=(float*)malloc(n*n*sizeof(float));
    c=(float*)malloc(n*n*sizeof(float));
    for (int i=0; i<n*n; i++)
    {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    float  *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * n * sizeof(float));
    cudaMalloc((void**)&d_b, n * n * sizeof(float));
    cudaMalloc((void**)&d_c, n * n * sizeof(float));

    cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul(d_a, d_b, d_c, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float m_s = 0;
    cudaEventElapsedTime(&m_s, start, stop);

    cudaMemcpy(c, d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<c[n*n-1]<<" "<<m_s<<"\n";

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
       return 1;
    }



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
