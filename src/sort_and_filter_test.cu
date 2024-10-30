#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "sort_and_filter.cuh" // Assuming this header is the one provided

#define N 1000 // Example size, adjust as needed

int main() {
    // Seed random number generator
    std::srand(std::time(0));

    // Create large array of keys with repeated keys
    int h_keys[N];
    int h_indices[N];
    for (int i = 0; i < N; i++) {
        h_keys[i] = std::rand() % (N / 10 + 1); // Repeat keys for aggregation
        h_indices[i] = i;
    }

    // Transfer data to device
    thrust::device_vector<int> d_keys(h_keys, h_keys + N);
    thrust::device_vector<int> d_indices(h_indices, h_indices + N);

    // Sort keys and indices on the device
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

    // Allocate output vector for KeyOccurences on the device
    thrust::device_vector<KeyOccurences> d_output(N);
    KeyOccurences empty_element;
    empty_element.key = -1;
    for (int i = 0; i < MAX_OCCURENCES; i++) empty_element.occurrences[i] = -1;
    thrust::fill(d_output.begin(), d_output.end(), empty_element);

    // Launch kernel to aggregate indices by key
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    aggregateIndices<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), 
                                              thrust::raw_pointer_cast(d_indices.data()), 
                                              thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize(); // Ensure kernel has completed

    // Copy results back to host to print
    std::vector<KeyOccurences> h_output(N);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Print results (limited to first 10 for brevity)
    for (int i = 0; i < N && h_output[i].key != -1; i++) {
        const auto &element = h_output[i];
        std::cout << "Key: " << element.key << " -> Indices: ";
        for (int j = 0; j < MAX_OCCURENCES; j++) {
            if (element.occurrences[j] != -1) 
                std::cout << element.occurrences[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
