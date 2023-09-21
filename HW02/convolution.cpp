#include "convolution.h"
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    std::size_t h_size = m / 2;

    for (std::size_t x = 0; x < n; ++x)
    {
     for (std::size_t y = 0; y < n; ++y)
       {
         output[x * n + y] = 0.0;
         for (std::size_t i = 0; i < m; ++i)
           {
             for (std::size_t j = 0; j < m; ++j)
               {
                  std::size_t imageX = x + i - h_size;
                  std::size_t imageY = y + j - h_size;
                  if (imageX >= 0 && imageX < n && imageY >= 0 && imageY < n)
                         {
                          output[x * n + y] += image[imageX * n + imageY] * mask[i * m + j];
                         }
               }
            }
         }
      }
}

