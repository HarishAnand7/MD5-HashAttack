#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include <bits/stdc++.h>
#include "matmul.cuh"

#define BLOCK_SIZE 16

using namespace std;
using std::cout;

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    unsigned block_dim=atoi(argv[2]);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0, 10.0);
    float *A, *B, *C;
    A=(float*)malloc(n*n*sizeof(float));
    B=(float*)malloc(n*n*sizeof(float));
    C=(float*)malloc(n*n*sizeof(float));
    for (int i=0; i<n*n; i++)
    {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }


    /*
    float** A = new float*[n];
    float** B = new float*[n];
    float** C = new float*[n];

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n; i++) {
        // Declare a memory block
        A[i] = new float[n];
    }

    for (int i = 0; i < n; i++) {

        // Declare a memory block
        B[i] = new float[n];
    }

    for (int i = 0; i < n; i++) {

        // Declare a memory block
        C[i] = new float[n];
    }

    //Random matrix generation

    std::uniform_real_distribution<float> distribution1(-100.0, 100.0);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            A[i][j] = distribution1(gen);
        }

    }

    std::uniform_real_distribution<float> distribution2(-100.0, 100.0);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            B[i][j] = distribution2(gen);
        }

    }
    */
       unsigned int hA, wA, wB;
    hA = wA = wB = n;
    // Load A and B to the device
    float* Ad;
    int size = hA * wA * sizeof(float);
    cudaMalloc((void**)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    float* Bd;
    int size1 = wA * wB * sizeof(float);
    cudaMalloc((void**)&Bd, size1);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    // Allocate C on the device
    float* Cd;
    int size2 = hA * wB * sizeof(float);
    cudaMalloc((void**)&Cd, size2);
    // Compute the execution configuration *assuming*
    // the matrix dimensions are multiples of BLOCK_SIZE
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid( wB/dimBlock.x , hA/dimBlock.y );
    // Launch the device computation

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_2(A, B, C, n, block_dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float m_s = 0;
    cudaEventElapsedTime(&m_s, start, stop);

    cudaMemcpy(C, Cd, size2, cudaMemcpyDeviceToHost);
    // Print time taken in milliseconds
    cout << m_s << "\n";

    // Print the first and last elements of the output array
    cout << C[0] << "\n";
    cout << C[n-1]<< "\n";
    cout << C[n-1]<< "\n";
    // Read C from the device
    // Free global memory
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
       return 1;
    }

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    /*
    delete(A);
    delete(B);
    delete(C);
    */
    free(A);
    free(B);
    free(C);
    return 0;



}

                                       


