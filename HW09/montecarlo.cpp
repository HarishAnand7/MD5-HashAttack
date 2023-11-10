#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius)
{
        int count = 0;
#pragma omp parallel for simd reduction(+ : count)
        for (size_t i = 0; i < n; i++) {
                if (x[i] * x[i] + y[i] * y[i] < radius * radius)
                        count += 1;
        }
        return count;
}

/*
#ifdef SIMD
#pragma omp parallel for simd reduction(+ : count)
#else
#pragma omp parallel for reduction(+ : count)
#endif
*/
