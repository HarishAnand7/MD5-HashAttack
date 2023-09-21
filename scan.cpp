#include "scan.h"

void scan(const float *arr, float *output, std::size_t n)
{
   if (n==0)
      {
        return;
      }


    output[0]=arr[0];
    for (int i=1;i<n;i++)
    {
        output[i]=output[0]+arr[i];
    }



}
