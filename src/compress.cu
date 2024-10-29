#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#define N 1024

typedef int KeyType;
int main() {
    // Initialize host array with random keys
    KeyType h_keys[N];
    for (int i = 0; i < N; i++) {
        h_keys[i] = rand() % (N / 10); // Random keys, could be adjusted
    }

    // Transfer to device
    thrust::device_vector<KeyType> d_keys(h_keys, h_keys + N);

    // Sort the keys
    thrust::sort(d_keys.begin(), d_keys.end());

    // Unique keys (this will compress the sequence)
    thrust::device_vector<KeyType> d_unique_keys = d_keys;
    auto end = thrust::unique(d_unique_keys.begin(), d_unique_keys.end());

    // Resize the vector to remove extra elements
    d_unique_keys.resize(thrust::distance(d_unique_keys.begin(), end));

    // Allocate vector for counts
    thrust::device_vector<int> d_counts(d_unique_keys.size());

    // Count occurrences
    thrust::counting_iterator<int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + d_unique_keys.size(),
                      d_counts.begin(),
                      [d_keys_ptr = thrust::raw_pointer_cast(d_keys.data()),
                       num_elements = d_keys.size()] __device__(int i) {
                          return thrust::count(thrust::device, d_keys_ptr, d_keys_ptr + num_elements, d_unique_keys[i]);
                      });

    // Copy back to host to print results
    std::vector<KeyType> h_unique_keys(d_unique_keys.size());
    std::vector<int> h_counts(d_counts.size());

    thrust::copy(d_unique_keys.begin(), d_unique_keys.end(), h_unique_keys.begin());
    thrust::copy(d_counts.begin(), d_counts.end(), h_counts.begin());

    // Print results
    for (size_t i = 0; i < h_unique_keys.size(); i++) {
        printf("Key: %d, Count: %d\n", h_unique_keys[i], h_counts[i]);
    }

    return 0;
}
