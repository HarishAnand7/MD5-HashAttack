#include <thrust/device_vector.h>
#include "count.cuh"

void count(const thrust::device_vector<int> &d_in,thrust::device_vector<int> &values,thrust::device_vector<int> &counts)
{

        thrust::device_vector<int> dA(int(d_in.size()));
        thrust::fill(dA.begin(), dA.end(), 1);
        values = d_in;
        thrust::sort(values.begin(), values.end());
        auto output = thrust::reduce_by_key(values.begin(), values.end(), dA.begin(), values.begin(), counts.begin());
        values.resize(output.first - values.begin());
        counts.resize(output.second - counts.begin());
}
