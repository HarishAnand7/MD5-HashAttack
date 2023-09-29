#include "vscale.cu"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{

   int N = std::atoi(argv[1]);

   float *hA=new float[N];
   float *hB=new float[N];

   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<float> dist_a(-10.0, 10.0);
   std::uniform_real_distribution<float> dist_b(0.0, 1.0);

    float *a, *b;

    int size= N*sizeof(float);

    cudaMalloc((void**)&a, size);
    cudaMalloc((void**)&b, size);

    for (int i = 0; i < N; i++)
    {
        hA[i] = dist_a(gen);
        hB[i] = dist_b(gen);
    }

    cudaMemcpy(a,hA,size,cudaMemcpyHostToDevice);
    cudaMemcpy(b,hB,size,cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vscale<<<numBlocks , blockSize>>>(a,b,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float secs = 0.0;
    cudaEventElapsedTime(&secs, start, stop);

    cudaMemcpy(hB,b,size,cudaMemcpyDeviceToHost);


    std::cout << secs << "\n";
    std::cout << hB[0] << "\n";
    std::cout << hB[N - 1] << "\n";


    cudaError_t cudaError = cudaGetLastError();
      if (cudaError != cudaSuccess)
      {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
      }

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);

    delete[] hA;
    delete[] hB;

    return 0;
}
