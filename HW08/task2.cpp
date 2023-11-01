#include "convolution.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
using namespace std;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char *argv[])
{
        std::size_t n = std::atoi(argv[1]); //matrix n*n
        int t = std::atoi(argv[2]); //threads
        size_t m = 3;  //mask size
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist1(-10.0f, 10.0f);
        std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);

        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;


        float *image=(float*)malloc( n* n* sizeof(float));
        float *mask=(float*)malloc( m* m* sizeof(float));

        for(size_t i=0;i<n*n;i++)
        {
           image[i]=dist1(gen);

        }

        for(size_t j=0;j<m*m;j++)
        {
           mask[j]=dist2(gen);
        }

        float *output=(float*)malloc( n* n* sizeof(float));

        omp_set_num_threads(t);
        start = high_resolution_clock::now();
        convolve(image, output, n , mask, m);
        end = high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);


        std::cout << output[0] <<"\t \t"<<output[n*n-1]<<"\t \t"<<duration.count() * 1000.0 << std::endl;

        free(image);
        free(output);
        free(mask);

        return 0;

}
