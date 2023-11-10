#include <chrono>
#include <cstring>
#include <random>
#include <algorithm>
#include <random>
#include <stdio.h>
#include "cluster.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char *argv[])
 {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    const float min = 0, max = n; // The range for the random number generator is 0 to n
    uniform_real_distribution<float> dist(min, max);


        float *arr = new float[n];
        float *centers = new float[t];
        float *dists = new float[t];

        for (int i = 0; i < n; i++)
        {
                arr[i] = dist(generator);
        }
        std::sort(arr, arr + n);

        for (int i = 1; i <= t; i++)
        {
                centers[i-1] = (2.0 * i - 1) * n / (2.0 * t);
                dists[i-1] = 0.0;
        }
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;


        omp_set_num_threads(t);
        start = high_resolution_clock::now();
        cluster(n, t, arr, centers, dists);
        end = high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        // looking for the largest elements in the array
        float mVal= dists[0];
        int p = 0;
        for (int i = 1; i < t; i++)
        {
                if (dists[i] > mVal)
                {
                        mVal = dists[i];
                        p = i;
                }
        }
        printf("%f\n%d\n%f\n", mVal, p , duration.count() * 1000.0);
        delete[] arr;
        delete[] centers;
        delete[] dists;
        return 0;
}
