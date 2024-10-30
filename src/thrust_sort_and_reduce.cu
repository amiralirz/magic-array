#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <iostream>

#define N 10                // size of input array
#define MAX_OCCURRENCES 20  // Max occurrences of each key

struct KeyValuePair {
    int key;
    int value_index;  // Store index of value instead of value itself
};

typedef struct {
    int key;
    int indices[MAX_OCCURRENCES];  // Indices of values
} AggregatedElement;

// a kernel to gather around all the indices of a given key
// each thread will gather all indices of it's own key
__global__ void aggregateIndices(int *keys, int *indices, AggregatedElement *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int key = keys[idx];
        int index = indices[idx];

        for (int i = 0; i < size; i++) {
            if (atomicCAS(&output[i].key, -1, key) == -1 || output[i].key == key) {
                for (int j = 0; j < MAX_OCCURRENCES; j++) {
                    if (atomicCAS(&output[i].indices[j], -1, index) == -1) {
                        return;
                    }
                }
            }
        }
    }
}

int main() {
    int h_keys[N] = {2, 3, 3, 5, 19, 2, 3, 5, 5, 19};
    int h_indices[N];
    for (int i = 0; i < N; i++) {
        h_indices[i] = i;
    }

    // HD
    thrust::device_vector<int> d_keys(h_keys, h_keys + N);
    thrust::device_vector<int> d_indices(h_indices, h_indices + N);

    // Sort by keys
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

    // Allocate output vector
    thrust::device_vector<AggregatedElement> d_output(N);
    AggregatedElement empty_element;
    empty_element.key = -1;
    for (int i = 0; i < MAX_OCCURRENCES; i++) empty_element.indices[i] = -1;

    thrust::fill(d_output.begin(), d_output.end(), empty_element);

    // Launch kernel to aggregate indices
    aggregateIndices<<<(N + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_indices.data()), thrust::raw_pointer_cast(d_output.data()), N);

    // DH
    std::vector<AggregatedElement> h_output(N);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Print results
    for (const auto &element : h_output) {
        if (element.key == -1) break;
        std::cout << "Key: " << element.key << " -> Indices: ";
        for (int i = 0; i < MAX_OCCURRENCES; i++) {
            if (element.indices[i] != -1) std::cout << element.indices[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
