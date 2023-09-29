#include "vscale.cuh"
__global__
void vscale(const float *a, float *b, unsigned int n)
{
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
  {
     b[index]=a[index]*b[index];
  }

}

