#include "stencil.cuh"
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Define the stencil_kernel here
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_memory[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    // Load data into shared memory
    for (int i = lane; i < 2 * static_cast<int>(R) + 1; i += blockDim.x) {
        shared_memory[i] = mask[i];
    }

    __syncthreads();

    if (tid < n) {
        float result = 0.0f;
        for (int i = -static_cast<int>(R); i <= static_cast<int>(R); ++i) {
            int idx = tid + i;
            if (idx >= 0 && idx < n) {
                result += image[idx] * shared_memory[i + R];
            }
        }
        output[tid] = result;
    }
}

// Define the stencil function here
void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    float* d_image;
    float* d_mask;
    float* d_output;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Define the execution configuration
    dim3 blockDim(threads_per_block);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the stencil kernel
    stencil_kernel<<<gridDim, blockDim, (2 * R + 1) * sizeof(float)>>>(d_image, d_mask, d_output, n, R);

    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back from device to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cout<<output[n-1]<<" ";

    // Clean up
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Print the time taken to execute the stencil function in milliseconds
    cout<<milliseconds<<endl;
}
