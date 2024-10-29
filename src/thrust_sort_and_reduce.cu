#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <iostream>

#define N 10
#define B 4  // Max occurrences of each key

struct KeyValuePair {
    int key;
    int value;
};

typedef struct {
    int key;
    int values[B];
} AggregatedElement;

__global__ void aggregateValues(int *keys, int *values, AggregatedElement *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int key = keys[idx];
        int value = values[idx];

        // Traverse the output array to find the right slot to aggregate
        for (int i = 0; i < size; i++) {
            if (atomicCAS(&output[i].key, -1, key) == -1 || output[i].key == key) {
                for (int j = 0; j < B; j++) {
                    if (atomicCAS(&output[i].values[j], -1, value) == -1) {
                        return;
                    }
                }
            }
        }
    }
}

int main() {
    // Example keys and values
    int h_keys[N] = {2, 3, 3, 5, 19, 2, 3, 5, 5, 19};
    int h_values[N] = {4, 1, 2, 0, 3, 1, 1, 2, 1, 2};

    // Transfer to device
    thrust::device_vector<int> d_keys(h_keys, h_keys + N);
    thrust::device_vector<int> d_values(h_values, h_values + N);

    // Sort by keys
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    // Allocate output vector
    thrust::device_vector<AggregatedElement> d_output(N);
    AggregatedElement empty_element;
    empty_element.key = -1;
    for (int i = 0; i < B; i++) empty_element.values[i] = -1;

    thrust::fill(d_output.begin(), d_output.end(), empty_element);

    // Launch kernel to aggregate values
    aggregateValues<<<(N + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_output.data()), N);

    // Copy back to host to print results
    std::vector<AggregatedElement> h_output(N);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Print results
    for (const auto &element : h_output) {
        if (element.key == -1) break;
        std::cout << "Key: " << element.key << " -> Values: ";
        for (int i = 0; i < B; i++) {
            if (element.values[i] != -1) std::cout << element.values[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
