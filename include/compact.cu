#include <cuda_runtime.h>
#include <thrust/device_vector.h>

struct KeyOccurrences {
    int key;
    int occurrences[10];  // Assuming max 10 occurrences per key for simplicity
    int count;
};

// Kernel to initialize KeyOccurrences array with empty values
__global__ void initKeyOccurrences(KeyOccurrences* d_compacted, int unique_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < unique_count) {
        d_compacted[idx].key = -1;  // Mark as empty
        for (int i = 0; i < 10; ++i) {
            d_compacted[idx].occurrences[i] = -1; // Initialize all as -1
        }
        d_compacted[idx].count = 0;
    }
}

// Kernel to compact sorted keys and indices into KeyOccurrences structure
__global__ void compactKeys(int* d_keys, int* d_indices, KeyOccurrences* d_compacted, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Atomic index tracking the position in compacted array
    __shared__ int unique_key_count;
    if (threadIdx.x == 0) unique_key_count = 0;
    __syncthreads();

    int current_key = d_keys[idx];
    int current_index = d_indices[idx];

    if (idx == 0 || current_key != d_keys[idx - 1]) {
        int pos = atomicAdd(&unique_key_count, 1); // New entry for a unique key
        d_compacted[pos].key = current_key;
        d_compacted[pos].occurrences[0] = current_index;
        d_compacted[pos].count = 1;
    } else {
        int pos = unique_key_count - 1;  // Use the last unique key entry
        int occ_pos = atomicAdd(&d_compacted[pos].count, 1);
        if (occ_pos < 10) { // Check max occurrences per key
            d_compacted[pos].occurrences[occ_pos] = current_index;
        }
    }
}
