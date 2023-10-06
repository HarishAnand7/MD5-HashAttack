#include <iostream>
#include <cuda.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <bits/stdc++.h>
#include "stencil.cuh"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char** argv)
{
    //duration<double, std::milli> duration_sec;

    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int R_t = 2*R+1;
    unsigned int threads_per_block = std::atoi(argv[3]);

    
    // Allocate and initialize arrays a and b with random values
    float* image = new float[n];
    float* output = new float[n];
    float* mask = new float[R_t];

    //Random matrix generation
    default_random_engine gen;
    uniform_real_distribution<float> distribution1(-1.0, 1.0);
    uniform_real_distribution<float> distribution2(-1.0, 1.0);

    for (unsigned int i=0; i<=n; i++)
    {
        image[i] = distribution1(gen);
    }


    for (unsigned int i=0; i<=R_t; i++)
    {
        mask[i] = distribution2(gen);
    }

    // Call the stencil function
    stencil(image, mask, output, n, R, threads_per_block);

    // Print the last element of the resulting output array
    // std::cout << output[n - 1] << std::endl;

    // Clean up
    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}
