#include "matmul.cuh"
#include <cuda_runtime.h>
const int sm=1<<10;
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

   __shared__ T s_a[sm];
   __shared__ T s_b[sm];

  int tmp = 0;

  for (int i = 0; i < n; i += blockDim.x)
  {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =B[i * n + threadIdx.y * n + col];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++)
    {
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  C[row * n + col] = tmp;
}


__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,unsigned int block_dim)
{
        int sm=block_dim*block_dim*sizeof(int);
        int Blocks= (n+block_dim-1 )/block_dim;
        dim3 dimBlock(block_dim, block_dim);
        dim3 dimGrid(Blocks, Blocks);
        matmul_kernel<<<dimGrid,dimBlock,sm>>>(A,B,C,n);
        cudaDeviceSynchronize();
}
__host__ void matmul_2(const float  *A, const float  *B, float *C, unsigned int n,unsigned int block_dim)
{
        int sm=block_dim*block_dim*sizeof(float);
        int Blocks= (n+block_dim-1 )/block_dim;
        dim3 dimBlock(block_dim, block_dim);
        dim3 dimGrid(Blocks, Blocks);
        matmul_kernel<<<dimGrid,dimBlock,sm>>>(A,B,C,n);
        cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double  *B, double *C, unsigned int n,unsigned int block_dim)
{
        int sm=block_dim*block_dim*sizeof(double);
        int Blocks= (n+block_dim-1 )/block_dim;
        dim3 dimBlock(block_dim, block_dim);
        dim3 dimGrid(Blocks, Blocks);
        matmul_kernel<<<dimGrid,dimBlock,sm>>>(A,B,C,n);
        cudaDeviceSynchronize();
}

