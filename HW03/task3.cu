#include <iostream>
#include <cuda.h>
#include <random>
#include "vscale.cuh"
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <bits/stdc++.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char** argv)
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    unsigned int n = atoi(argv[1]);

    
    float* a = new float[n];
    float* b = new float[n];

    
    default_random_engine gen;
    uniform_real_distribution<float> distribution1(-10.0, 10.0);
    uniform_real_distribution<float> distribution2(0.0, 1.0);

    for (int i; i<=n; i++)
    {
        a[i] = distribution1(gen);
        b[i] = distribution2(gen);
    }

    float* d_a;
    float* d_b;

    cudaMalloc((void**)&d_a, n*sizeof(float));
    cudaMalloc((void**)&d_b, n*sizeof(float));

    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockDim(16);  
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);  


    
    auto start_time = high_resolution_clock::now();
    
    vscale<<<gridDim, blockDim>>>(d_a, d_b, n);
    auto end_time = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_time - start_time);
    
    cout << duration_sec.count() << " ";

    
    cout << b[0] << " ";
    cout << b[n-1] << endl;

    
    delete[] a;
    delete[] b;
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}

