#include "msort.h"
#include <algorithm>


void ReccurMsort(int *arr, const std::size_t n, const std::size_t threshold, int TH)
{
        if (n <= 1)
        {
                return;
        }
        if (n < threshold)
        {
                for (auto i = arr; i != arr + n; i++)
                {

                        std::rotate(std::upper_bound(arr, i, *i), i, i+1);
                }
                return;
        }

        if (TH == 1)
        {
                ReccurMsort(arr, n/2, threshold, TH);
                ReccurMsort(arr+n/2, n - n/2, threshold, TH);
        }
        else
        {
        #pragma omp task
                ReccurMsort(arr, n/2, threshold, TH/2);
        #pragma omp task
                ReccurMsort(arr+n/2, n- n/2, threshold, TH - TH/2);
        #pragma omp taskwait
        }
        std::inplace_merge(arr, arr + n/2, arr + n);
}
void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
#pragma omp parallel
#pragma omp single
        ReccurMsort(arr, n, threshold, omp_get_num_threads());
}
