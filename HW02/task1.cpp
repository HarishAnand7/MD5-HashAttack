#include <stdio.h>
#include <ctime>
#include "scan.h"
#include <iostream>
#include <chrono>
#include <cmath>

using std::chrono::duration;
int main(int argc, char *argv[]) {
     std::chrono::high_resolution_clock::time_point start;
     std::chrono::high_resolution_clock::time_point end;
     std::chrono::duration<double, std::milli> duration_sec;

    int n = std::atoi(argv[1]);

    std::srand(static_cast<unsigned>(std::time(nullptr)));


    float *arr = new float[n];
    for (int i = 0; i < n; ++i)
    {
        arr[i] = static_cast<float>(std::rand()) /  static_cast<float>(RAND_MAX) * 2.0 - 1.0;
    }


    float *output = new float[n];


    start = std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    end =std::chrono::high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end-start);


    std::cout << duration_sec.count()<<"\n" ;


    std::cout << output[0]<<"\n"  ;
    std::cout << output[n - 1]<<"\n" ;


    delete[] arr;
    delete[] output;

    return 0;
}

