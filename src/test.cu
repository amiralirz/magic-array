#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include "smart_array.cuh"
#include "peek.cuh"

#define N 32   // test input size

int main() {
    keytype* h_keys = (keytype*)malloc(N * sizeof(keytype));

    srand(0);
    // Initialize the input and sorted arrays
    for (int i = 0; i < N; i++) {
        h_keys[i] = rand() % (N / 2); // limiting the values to ensure key repetition
    }
    keytype* d_keys;
    int* d_indices;

    // --------------------------- allocating GPU memory ---------------------------
    cudaMalloc(&d_keys, N * sizeof(keytype));
    cudaMalloc(&d_indices, N * sizeof(valuetype));

    // --------------------------- Moving data from RAM to GPU memory ---------------------------
    cudaMemcpy(d_keys, h_keys, N * sizeof(keytype), cudaMemcpyHostToDevice);

    MagicArray arr(10000);
    arr.insert(d_keys, d_indices, N);
    // arr.printTable();

    free(h_keys);
    cudaFree(d_keys);
    cudaFree(d_indices);

    return 0;
}
