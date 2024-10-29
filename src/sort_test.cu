#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "compact.cu"  // Include compact functions

struct key_index_pair {
    int key;
    int index;
    __host__ __device__ key_index_pair(int k = 0, int i = 0) : key(k), index(i) {}
};

struct KeyComparator {
    __host__ __device__ bool operator()(const key_index_pair& a, const key_index_pair& b) const {
        return a.key < b.key;
    }
};

// Functor to extract keys
struct ExtractKey {
    __host__ __device__ int operator()(const key_index_pair& p) const {
        return p.key;
    }
};

// Functor to extract indices
struct ExtractIndex {
    __host__ __device__ int operator()(const key_index_pair& p) const {
        return p.index;
    }
};

int main() {
    int h_keys[] = {5, 3, 3, 19, 2};
    int N = sizeof(h_keys) / sizeof(h_keys[0]);

    thrust::host_vector<key_index_pair> h_pairs(N);
    for (int i = 0; i < N; ++i) {
        h_pairs[i] = key_index_pair(h_keys[i], i);
    }
    thrust::device_vector<key_index_pair> d_pairs = h_pairs;

    thrust::sort(d_pairs.begin(), d_pairs.end(), KeyComparator());

    thrust::device_vector<int> d_keys(N), d_indices(N);
    thrust::transform(d_pairs.begin(), d_pairs.end(), d_keys.begin(), ExtractKey());
    thrust::transform(d_pairs.begin(), d_pairs.end(), d_indices.begin(), ExtractIndex());

    int unique_count = N;
    KeyOccurrences* d_compacted;
    cudaMalloc(&d_compacted, unique_count * sizeof(KeyOccurrences));
    initKeyOccurrences<<<(unique_count + 255) / 256, 256>>>(d_compacted, unique_count);

    compactKeys<<<(N + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_keys.data()),
                                          thrust::raw_pointer_cast(d_indices.data()), 
                                          d_compacted, N);

    KeyOccurrences h_compacted[unique_count];
    cudaMemcpy(h_compacted, d_compacted, unique_count * sizeof(KeyOccurrences), cudaMemcpyDeviceToHost);

    std::cout << "Compacted results:\n";
    for (int i = 0; i < unique_count; ++i) {
        if (h_compacted[i].count > 0) {
            std::cout << "(" << h_compacted[i].key << ", [";
            for (int j = 0; j < h_compacted[i].count; ++j) {
                if (h_compacted[i].occurrences[j] != -1)
                    std::cout << h_compacted[i].occurrences[j];
                if (j < h_compacted[i].count - 1) std::cout << ", ";
            }
            std::cout << "])\n";
        }
    }

    cudaFree(d_compacted);
    return 0;
}
