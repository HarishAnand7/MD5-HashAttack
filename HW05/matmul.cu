#include "matmul.cuh"
// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
#define BLOCK_SIZE 16
__global__ void matmul_kernel(const float *A, const float *B, float *C,  unsigned int n)
{
int wA,wB;
wA=wB=n;
// Block index
int bx = blockIdx.x; //the B (and C) matrix sub-block column index
int by = blockIdx.y; //the A (and C) matrix sub-block row index
// Thread index
int tx = threadIdx.x; //the column index in the sub-block
int ty = threadIdx.y; //the row index in the sub-block
// Index of the first sub-matrix of A processed by the block
int aBegin = wA * BLOCK_SIZE * by;
// Index of the last sub-matrix of A processed by the block
int aEnd = aBegin + wA - 1;
// Step size used to iterate through the sub-matrices of A
int aStep = BLOCK_SIZE;
// Index of the first sub-matrix of B processed by the block
int bBegin = BLOCK_SIZE * bx;
// Step size used to iterate through the sub-matrices of B
int bStep = BLOCK_SIZE * wB;
// The element of the block sub-matrix that is computed
// by the thread
float Csub = 0;
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
// Loop over all the sub-matrices (tiles) of A and B required to
// compute the block sub-matrix; moving in A left to right in
// a row, and in B from top to bottom in a column
for (int a = aBegin, b = bBegin;a <= aEnd;a += aStep, b += bStep)
{
// Load tiles from global memory into shared memory; each
// thread loads one element of the two tiles from A & B
As[ty][tx] = A[a + wA * ty + tx];
Bs[ty][tx] = B[b + wB * ty + tx];
// Synchronize to make sure the matrices are loaded
__syncthreads();
// Each thread in this block computes one element
// of the block sub-matrix (tile). Thread with indexes
// ty and tx computes in this tile the entry [ty][tx].
for (int k = 0; k < BLOCK_SIZE; ++k)
Csub += As[ty][k] * Bs[k][tx];
// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
__syncthreads();
}
// Write the block sub-matrix to global memory;
// each thread writes one element
int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB * ty + tx] = Csub;

}

__host__ void matmul_2(const float *A, const float *B, float *C , unsigned int n, unsigned int block_dim )
{

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid( n/dimBlock.x , n/dimBlock.y );

    matmul_kernel<<<dimGrid, dimBlock>>>(A, B, C, n );


}


                                                                                                                                     
