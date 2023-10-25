#include <stdio.h>
#include <cuda.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
using namespace std;
using std::cout;

int main(int argc, char *argv[])
{
        int n = atoi(argv[1]);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist1(-1.0f, 1.0f);


        thrust::host_vector<float> hA(n);
        for (int i = 0; i < n; i++)
        {
                hA[i] = dist1(gen);
        }
        thrust::device_vector<float> dA = hA;
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        float result = thrust::reduce(dA.begin() , dA.end(),0.0,thrust::plus<float>());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
         return 1;
        }

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << result << std::endl;
        std::cout << ms << std::endl;

        return 0;
}

