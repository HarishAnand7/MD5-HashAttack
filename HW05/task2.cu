#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include <bits/stdc++.h>
#include "reduce.cuh"

#define BLOCK_SIZE 16

using namespace std;
using std::cout;

int main(int argc, char *argv[])
{
    unsigned int n = atoi(argv[1]);
    unsigned threads_per_block=atoi(argv[2]);
    float *A,R;
    A=(float*)malloc(n*sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(unsigned int i=0;i<n;i++)
    {
         A[i]=dist(gen);
    }

    float *input;
    cudaMalloc((void**)&input, n*sizeof(float));
    cudaMemcpy(input, A, n*sizeof(float), cudaMemcpyHostToDevice);

    float *output;
    cudaMalloc((void**)&output, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce(&input,&output,n,threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float m_s = 0;
    cudaEventElapsedTime(&m_s, start, stop);

    cudaMemcpy(&R,output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<R<<"\n"<<m_s<<"\n";

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
       return 1;
    }

    cudaFree(input);
    cudaFree(output);
    free(A);
}
