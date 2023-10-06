#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    float temp=0;
    int sn= n;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row= tid/sn;
    int col= tid%sn;

    if(tid < sn*sn)

    {

    for (int i = 0; i < sn; ++i)
    {
        temp += A[row * sn + i] * B[i * sn + col];
    }

    C[row * sn + col] = temp;


    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    int Blocks= n*n / threads_per_block;
    if ((n*n) % threads_per_block !=0)
    {
            Blocks =Blocks+1;
    }
    matmul_kernel<<<Blocks, threads_per_block>>>(A, B, C, n);
}
