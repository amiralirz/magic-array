#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "sort_and_filter.cuh"

int main() {
    KeyValuePair h_input[N];
    SortedElement h_sorted_arr[N];

    srand(0);
    // Initialize the input and sorted arrays
    for (int i = 0; i < N; i++) {
        h_input[i].key = rand();
        sprintf(h_input[i].payload, "payload %d", h_input[i].key);
        h_sorted_arr[i].key = -1;
        for (int j = 0; j < B; j++) {
            h_sorted_arr[i].occurrences[j] = -1;
        }
    }

    for (int i=0; i<N/2; i++) h_input[i].key = h_input[N - i - 1].key;
    for (int i=0; i<N; i++) printf("%d\n", h_input[i].key);

    KeyValuePair *d_input;
    SortedElement *d_sorted_arr;
    int *d_keys, *d_indices;

    cudaMalloc(&d_input, N * sizeof(KeyValuePair));
    cudaMalloc(&d_sorted_arr, N * sizeof(SortedElement));
    cudaMalloc(&d_keys, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(KeyValuePair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_arr, h_sorted_arr, N * sizeof(SortedElement), cudaMemcpyHostToDevice);

    extractKeys<<<(N + 255) / 256, 256>>>(d_input, d_keys, d_indices, N);

    sortKeys(d_keys, d_indices, N);

    buildSortedArray<<<(N + 255) / 256, 256>>>(d_keys, d_indices, d_sorted_arr, N);

    cudaMemcpy(h_sorted_arr, d_sorted_arr, N * sizeof(SortedElement), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        if (h_sorted_arr[i].key == -1) break;
        printf("Key: %d -> ", h_sorted_arr[i].key);
        for (int j = 0; j < B; j++) {
            if (h_sorted_arr[i].occurrences[j] != -1) {
                printf("%d ", h_sorted_arr[i].occurrences[j]);
            }
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_sorted_arr);
    cudaFree(d_keys);
    cudaFree(d_indices);

    return 0;
}
