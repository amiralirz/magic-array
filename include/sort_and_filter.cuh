// sort_and_filter.cu
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>

// Maximum number of occurrences per key
const int b = 5;

// Structure for each input element
struct key_value_pair {
    int key;
    char payload[150];  // Example payload
};

// Structure for each entry in the sorted array
struct sorted_entry {
    int key;
    int occurrences[b];  // Array to store indices of occurrences
};

// Comparator for sorting based on `key`
struct KeyComparator {
    __host__ __device__ bool operator()(const key_value_pair& a, const key_value_pair& b) const {
        return a.key < b.key;
    }
};

// CUDA kernel to populate sorted_arr with unique keys and their occurrences
__global__ void populate_sorted_arr(const key_value_pair* d_input, sorted_entry* d_sorted, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    int key = d_input[idx].key;
    int occurrence_index = 0;

    // Only process the first occurrence of each unique key
    if (idx == 0 || d_input[idx].key != d_input[idx - 1].key) {
        // Initialize a new entry in `sorted_arr`
        d_sorted[idx].key = key;
        for (int i = 0; i < b; ++i) {
            d_sorted[idx].occurrences[i] = -1;  // Initialize to -1 (unused slots)
        }

        // Find occurrences of the key and store the original indices
        for (int j = idx; j < N && d_input[j].key == key && occurrence_index < b; ++j) {
            d_sorted[idx].occurrences[occurrence_index++] = j; // Store original index
        }
    }
};

// Helper function to sort the input array by key using Thrust
void sort_input_by_key(key_value_pair* d_input, int N) {
    thrust::device_ptr<key_value_pair> dev_ptr(d_input);
    thrust::sort(dev_ptr, dev_ptr + N, KeyComparator());
};
