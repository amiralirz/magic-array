#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "sort_and_filter.cuh"

int main() {
    const int N = 100;  // Example size of the input array
    // key_value_pair h_input[N] = {
    //     {5, "payload"}, {3, "payload"}, {5, "payload"}, {2, "payload"}, {3, "payload"},
    //     {5, "payload"}, {4, "payload"}, {2, "payload"}, {1, "payload"}, {3, "payload"}
    // };
    srand(0);
    // key_value_pair* h_input = (key_value_pair*) malloc(sizeof(key_value_pair) * N);
    key_value_pair h_input[N];
    int key;
    for (int i=0; i<N; i++){
        key = rand();
        h_input[i].key = key; // Use dot notation to access struct members
        sprintf(h_input[i].payload, "payload %d", key);
    }
    for (int i=0; i<N/2; i++){
        h_input[i].key = h_input[N - i - 1].key;
    }
    for (int i=0; i<N; i++) printf("key: %d -> %s\n",h_input[i].key, h_input[i].payload);

    // Allocate device memory and copy input data to device
    key_value_pair* d_input;
    cudaMalloc(&d_input, N * sizeof(key_value_pair));
    cudaMemcpy(d_input, h_input, N * sizeof(key_value_pair), cudaMemcpyHostToDevice);

    // Sort the input array by keys
    sort_input_by_key(d_input, N);
    cudaDeviceSynchronize();

    // Allocate memory for the sorted array
    sorted_entry* d_sorted;
    cudaMalloc(&d_sorted, N * sizeof(sorted_entry));

    // Launch the kernel to populate sorted_arr
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    populate_sorted_arr<<<blocks, threadsPerBlock>>>(d_input, d_sorted, N);
    cudaDeviceSynchronize();

    // Copy results back to host and print them
    sorted_entry* h_sorted = new sorted_entry[N];
    cudaMemcpy(h_sorted, d_sorted, N * sizeof(sorted_entry), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (h_sorted[i].occurrences[0] != -1) {
            std::cout << "Key: " << h_sorted[i].key << " | Occurrences: ";
            for (int j = 0; j < b; ++j) {
                if (h_sorted[i].occurrences[j] != -1) {
                    std::cout << h_sorted[i].occurrences[j] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_sorted);
    delete[] h_sorted;

    return 0;
}
