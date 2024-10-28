#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include "sort_and_filter.cuh"
#include "peek.cuh"

#define N 128   // Array size

int main() {
    KeyValuePair h_input[N];
    KeyOccurences h_sorted_arr[N];

    srand(0);
    // Initialize the input and sorted arrays
    for (int i = 0; i < N; i++) {
        h_input[i].key = rand() % (N / 2); // limiting the values to ensure key repetition
        // std::cout<<h_input[i].key<<" ";
        h_sorted_arr[i].key = -1;
        for (int j = 0; j < B; j++) {
            h_sorted_arr[i].occurrences[j] = -1;
        }
    }
    // std::cout<<std::endl;
    KeyValuePair *d_input;
    KeyOccurences *d_sorted_arr;
    int *d_keys, *d_indices;

    // --------------------------- allocating GPU memory ---------------------------
    cudaMalloc(&d_input, N * sizeof(KeyValuePair));
    cudaMalloc(&d_sorted_arr, N * sizeof(KeyOccurences));
    cudaMalloc(&d_keys, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));

    // --------------------------- Moving data from RAM to GPU memory ---------------------------
    cudaMemcpy(d_input, h_input, N * sizeof(KeyValuePair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_arr, h_sorted_arr, N * sizeof(KeyOccurences), cudaMemcpyHostToDevice);

    extractKeys<<<(N + 255) / 256, 256>>>(d_input, d_keys, d_indices, N);
    // peekMemory(d_keys, N);
    // peekMemory(d_indices, N);

    sortKeys(d_keys, d_indices, N);
    // peekMemory(d_keys, N);

    buildSortedArray<<<(N + 255) / 256, 256>>>(d_keys, d_indices, d_sorted_arr, N);
    cudaDeviceSynchronize();
    peekMemory(d_sorted_arr, N);

    cudaFree(d_input);
    cudaFree(d_sorted_arr);
    cudaFree(d_keys);
    cudaFree(d_indices);

    return 0;
}
