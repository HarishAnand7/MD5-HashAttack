#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2 )+ threadIdx.x;

    float mySum= (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n)
    {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid]=mySum;

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (tid < stride) {
            sdata[tid]= mySum = mySum+sdata[tid + stride];
        }
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }



}

__host__ void reduce(float **input, float **output, unsigned int N,unsigned int threads_per_block)
{


 int sm=threads_per_block*sizeof(float);
 int Blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    *output = *input;

    while (Blocks > 1)
    {
        reduce_kernel<<<Blocks, threads_per_block, sm>>>(*input, *output, N);
        N = Blocks;
        Blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }

    cudaDeviceSynchronize();

}
