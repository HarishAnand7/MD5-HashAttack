#include <stdio.h>
#include <cuda.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include "count.cuh"
using namespace std;
using std::cout;


int main(int argc, char *argv[])
{
        int n = atoi(argv[1]);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist1(0, 500);


        thrust::host_vector<int> h_in(n);
        for (int i = 0; i < n; i++)
        {
                h_in[i] = dist1(gen);
        }
        thrust::device_vector<int> d_in = h_in;
        thrust::device_vector<int> values(n);
        thrust::device_vector<int> counts(n);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        count(d_in, values, counts);
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

        std::cout << values.back() << std::endl;
        std::cout << counts.back() << std::endl;
        std::cout << ms << std::endl;


        return 0;
}
