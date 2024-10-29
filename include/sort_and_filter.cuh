#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


#define N 100   // Array size
#define B 4      // Max occurrences of each key

typedef struct {
    int key;
    char payload[256];
} KeyValuePair;

typedef struct {
    int key;
    int occurrences[B];
} SortedElement;

__global__ void extractKeys(KeyValuePair *input, int *keys, int *indices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keys[idx] = input[idx].key;
        indices[idx] = idx;
    }
}

void sortKeys(int *d_keys, int *d_indices, int size) {
    thrust::device_ptr<int> d_keys_ptr(d_keys);
    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + size, d_indices_ptr);
}

__global__ void buildSortedArray(int *keys, int *indices, SortedElement *sorted_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int key = keys[idx];
        int index = indices[idx];

        for (int i = 0; i < size; i++) {
            if (sorted_arr[i].key == key) {
                for (int j = 0; j < B; j++) {
                    if (sorted_arr[i].occurrences[j] == -1) {
                        sorted_arr[i].occurrences[j] = index;
                        return;
                    }
                }
            }
        }

        for (int i = 0; i < size; i++) {
            if (sorted_arr[i].key == -1) {
                sorted_arr[i].key = key;
                sorted_arr[i].occurrences[0] = index;
                return;
            }
        }
    }
}
