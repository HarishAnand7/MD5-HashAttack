#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
#include <cuda.h>
#include <random>
using namespace std;
using std::cout;
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


int main(int argc, char *argv[])
{
        int n = atoi(argv[1]);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist1(-1.0f, 1.0f);

        float *h_in = new float[n];
        for (int i = 0; i < n; i++)
        {
                h_in[i] = dist1(gen);
        }

        float* d_in = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(float)*n));
        CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float)*n, cudaMemcpyHostToDevice));
        float* d_sum = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(float) * 1));

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float result;
        CubDebugExit(cudaMemcpy(&result, d_sum, sizeof(float), cudaMemcpyDeviceToHost));


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

       if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
       if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
       if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

       return 0;
}

