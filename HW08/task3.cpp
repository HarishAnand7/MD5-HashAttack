#include "msort.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <cassert>
#include <stdio.h>

using namespace std;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char *argv[])
{
        int n = atol(argv[1]);
        int t = atol(argv[2]);
        int ts = atol(argv[3]);
        std::random_device entropy_source;
        std::mt19937_64 generator(entropy_source());
        const int min = -1000, max = 1000;
        std::uniform_int_distribution<int> dist1(min,max);

        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;

        int *arr= new int[n];

        for(int i=0;i<n;i++)
        {
                arr[i]=dist1(generator);
        }


        omp_set_num_threads(t);
        omp_set_nested(1);

        start = high_resolution_clock::now();
        msort(arr,n,ts);
        end = high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);


        std::cout << arr[0]<<std::endl;
        std::cout<<arr[n-1]<<std::endl;
        std::cout<<duration.count() * 1000.0 << std::endl;
        assert(std::is_sorted(arr, arr+n));
        delete [] arr;


}
