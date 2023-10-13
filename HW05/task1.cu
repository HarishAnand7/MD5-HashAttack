#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include <bits/stdc++.h>
#include "matmul.cuh"


using namespace std;
using std::cout;


int main(int argc, char* argv[])
{

    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim =std::atoi(argv[2]);
    int* h_Ai = new int[n * n];
    int* h_Bi = new int[n * n];

    float* h_Af = new float[n * n];
    float* h_Bf = new float[n * n];

    double* h_Ad = new double[n * n];
    double* h_Bd = new double[n * n];

    int* h_C1 = new int[n * n];
    float* h_C2 = new float[n * n];
    double* h_C3 = new double[n * n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist1(-10, 10);
    std::uniform_real_distribution<float> dist2(-10.0f, 10.0f);
    std::uniform_real_distribution<double> dist3(-10.0, 10.0);

    for (unsigned int i = 0; i < n * n; ++i)
    {
        h_Ai[i] = dist1(gen);
        h_Bi[i] = dist1(gen);
    }
    for (unsigned int i = 0; i < n * n; ++i)
    {
        h_Af[i] = dist2(gen);
        h_Bf[i] = dist3(gen);
    }

    for (unsigned int i = 0; i < n * n; ++i)
    {
        h_Ad[i] = dist3(gen);
        h_Bd[i] = dist3(gen);
    }


    int* d_Ai, *d_Bi, *d_C1;
    float* d_Af, *d_Bf, *d_C2;
    double* d_Ad, *d_Bd, *d_C3;
    cudaMalloc((void**)&d_Ai, n * n * sizeof(int));
    cudaMalloc((void**)&d_Bi, n * n * sizeof(int));

    cudaMalloc((void**)&d_Af, n * n * sizeof(float));
    cudaMalloc((void**)&d_Bf, n * n * sizeof(float));

    cudaMalloc((void**)&d_Ad, n * n * sizeof(double));
    cudaMalloc((void**)&d_Bd, n * n * sizeof(double));

    cudaMalloc((void**)&d_C1, n * n * sizeof(int));
    cudaMalloc((void**)&d_C2, n * n * sizeof(float));
    cudaMalloc((void**)&d_C3, n * n * sizeof(double));

    cudaMemcpy(d_Ai, h_Ai, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bi, h_Bi, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Af, h_Af, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bf, h_Bf, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Ad, h_Ad, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bd, h_Bd, n * n * sizeof(double), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1(d_Ai, d_Bi, d_C1, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);


    cudaEventRecord(start);
    matmul_2(d_Af, d_Bf, d_C2, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);


    cudaEventRecord(start);
    matmul_3(d_Ad, d_Bd, d_C3, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start, stop);


    cudaMemcpy(h_C1, d_C1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << h_C1[0] << std::endl;
    std::cout << h_C1[n * n - 1] << std::endl;
    std::cout << milliseconds1 << std::endl;
    std::cout << h_C2[0] << std::endl;
    std::cout << h_C2[n * n - 1] << std::endl;
    std::cout << milliseconds2 << std::endl;
    std::cout << h_C3[0] << std::endl;
    std::cout << h_C3[n * n - 1] << std::endl;
    std::cout << milliseconds3 << std::endl;

    cudaFree(d_Ai);
    cudaFree(d_Bi);
    cudaFree(d_Af);
    cudaFree(d_Bf);
    cudaFree(d_Ad);
    cudaFree(d_Bd);

    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
    delete[] h_Ai;
    delete[] h_Bi;
    delete[] h_Af;
    delete[] h_Bf;
    delete[] h_Ad;
    delete[] h_Bd;


    delete[] h_C1;
    delete[] h_C2;
    delete[] h_C3;

    return 0;
}
