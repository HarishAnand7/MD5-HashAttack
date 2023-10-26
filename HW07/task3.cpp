#include <iostream>
#include <omp.h>
#include <stdio.h>

int factorial(int n)
{
    if(n>1)
       return n*factorial(n-1);
    else
       return 1;

}

int main()
{

        const int Threads = 4;
        omp_set_num_threads(Threads);

        printf("Number of threads: %d\n", Threads);
#pragma omp parallel
        {
                int tid = omp_get_thread_num();

                printf("I am thread No.: %d\n", tid);
        }

#pragma omp parallel for
        for (int i = 1; i <=8; i++)
        {
                printf("%d!=%d\n", i, factorial(i));
        }
        return 0;
}
