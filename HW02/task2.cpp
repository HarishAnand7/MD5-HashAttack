#include "convolution.h"
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    std::size_t n = std::atoi(argv[1]);
    std::size_t m = std::atoi(argv[2]);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;


    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    for (std::size_t i = 0; i < n * n; ++i)
    {
        image[i] = static_cast<float>(rand()) / RAND_MAX * 20.0 - 10.0;
    }

    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = static_cast<float>(rand()) / RAND_MAX * 2.0 - 1.0;
    }


    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();

    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << duration.count() * 1000.0 << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}

