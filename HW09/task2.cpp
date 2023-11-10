#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdio.h>
#include "montecarlo.h"
#include <stdio.h>
#include <stdlib.h>


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    float r = 1.0;

    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    const float min = -r, max = r;
    uniform_real_distribution<float> dist(min, max);

    float *x = new float[n];
    float *y = new float[n];

    for (int i = 0; i < n; i++)
   {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }


    double pi;
    double tot_time = 0.0;
    int avg_i = 10;
    for (int i = 0; i < avg_i; i++)
    {
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        pi = 4.0 * montecarlo(n, x, y, r) / n;
        double end =  omp_get_wtime();
        tot_time += (end - start) * 1000;
    }

    printf("%f\n%f\n", pi, tot_time/avg_i);
    delete[] x;
    delete[] y;
    return 0;

}

