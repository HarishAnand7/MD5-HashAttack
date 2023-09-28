#include<iostream>
#include<cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include<random>

__global__
void arrmat(int *dA,int a)
{
  int index=blockIdx.x * blockDim.x + threadIdx.x;

  int x=threadIdx.x;
  int y=blockIdx.x;

  dA[index]=a*x +y;

}


int main()
{
  int len=16;
  int *dA;
  int hA[len];

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(1, 10);
  int a= dist(gen);


  int size=sizeof(int)*len;

  cudaMalloc((void**)&dA,size);
  cudaMemset(dA, 0, size);

  arrmat<<<2 , 8 >>>(dA,a);


  cudaMemcpy(hA,dA,size,cudaMemcpyDeviceToHost);

  for(int i=0;i<len;i++)
  {
          std::printf("%d " ,hA[i]);
  }


  cudaError_t cudaError = cudaGetLastError();
     if (cudaError != cudaSuccess)
     {
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
       return 1;
     }

  cudaDeviceSynchronize();
  cudaFree(dA);

  return 0;



}
