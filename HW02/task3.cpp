#include "matmul.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char* argv[])
{

    unsigned int n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;

    for (unsigned int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    std::vector<void (*)(const double*, const double*, double*, const unsigned int)> mmul_functions =
    {
        mmul1, mmul2, mmul3
    };
    mmul_functions.push_back([](const double* A, const double* B, double* C, const unsigned int n)
    {
        std::vector<double> A_v(A, A + n * n);
        std::vector<double> B_v(B, B + n * n);
        mmul4(A_v, B_v, C, n);
    });

    std::cout << n <<"\n";

    for (unsigned int i = 0; i < mmul_functions.size(); i++)
    {
        auto mmul_function = mmul_functions[i];

        start = high_resolution_clock::now();
        mmul_function(A.data(), B.data(), C.data(), n);
        end = high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << duration.count() * 1000.0 << "\n";
        std::cout << C[n * n - 1] << "\n";
    }

    return 0;
}

