#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "magic_array.cuh"

const int numKeys = 130000;
const int arraySize = 1000; // Example size for the magic array

int main() {
    cudaError_t cudaStatus;
    long *sorted_keys;

    cudaStatus = cudaMallocManaged(&sorted_keys, numKeys * sizeof(long));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error allocating managed memory for sorted_keys: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Populate sorted_keys with random values or sorted test data
    for (int i = 0; i < numKeys; i++) {
        sorted_keys[i] = mrand48(); // Replace with actual sorted data
    }

    MagicArray magic_array(arraySize);

    int *d_insertion_pointer;
    cudaStatus = cudaMallocManaged(&d_insertion_pointer, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error allocating managed memory for d_insertion_pointer: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    *d_insertion_pointer = 0; // Start insertion pointer at 0

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    countOccurrences<<<numBlocks, blockSize>>>(sorted_keys, numKeys, magic_array.getKeysPointer(), magic_array.getCountsPointer(), arraySize, d_insertion_pointer);

    cudaDeviceSynchronize();

    printf("here\n");
    printf("magic_array.getCountsPointer() = %p\n", magic_array.getCountsPointer());
    printf("magic_array.getAllCounts() = %ld\n", magic_array.getAllCounts());
    // Optional: Print contents of magic_keys and magic_counts
    for (int i = 0; i < arraySize; ++i) {
        if (magic_array.getCountsPointer()[i] > 0) {
            std::cout << "Key: " << magic_array.getKeysPointer()[i] << ", Count: " << magic_array.getCountsPointer()[i] << std::endl;
        }
    }
    printf("there\n");


    // // Free memory
    // cudaFree(sorted_keys);
    // cudaFree(d_insertion_pointer);

    return 0;
}
