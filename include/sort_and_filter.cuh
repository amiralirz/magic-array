#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define B 4      // Max occurrences of each key

using keytype = int;

typedef struct {
    keytype key;
    char payload[256];
} KeyValuePair;

typedef struct{
    keytype key;
    int indx;
} KeyIndexPair;

typedef struct {
    keytype key;
    int occurenceCount = 0;
    int occurrences[B];
} KeyOccurences;


// this function copies the keys and their indices and writes them in seperate arrays
__global__ void extractKeys(KeyValuePair *input, int *keys, int *indices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keys[idx] = input[idx].key;
        indices[idx] = idx;
    }
}

// This functions sorts the keys and rearranges their indices respectively
void sortKeys(int *d_keys, int *d_indices, int size) { 
    thrust::device_ptr<keytype> d_keys_ptr(d_keys);
    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + size, d_indices_ptr);
}

__global__ void buildSortedArray(int *keys, int *indices, KeyOccurences *sorted_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keytype key = keys[idx];
        int index = indices[idx];

        // inserting to an existing bucekt
        for (int i = 0; i < size; i++) { // TODO : don't search using brute force
            if (sorted_arr[i].key == key) {
                for (int j = 0; j < B; j++) {
                    if (sorted_arr[i].occurrences[j] == -1) { // TODO : use atomice operations here
                        sorted_arr[i].occurrences[j] = index;
                        return;
                    }
                }
            }
        }

        // inserting to a new bucket if no exisiting buckets match
        for (int i = 0; i < size; i++) {
            if (sorted_arr[i].key == -1) {
                sorted_arr[i].key = key;
                for (int j = 0; j < B; j++) {
                    if (sorted_arr[i].occurrences[j] == -1) { // TODO : use atomice operations here
                        sorted_arr[i].occurrences[j] = index;
                        return;
                    }
                }
                return;
            }
        }
    }
}
