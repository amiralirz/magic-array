#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define MAX_OCCURENCES 50 // Max occurrences of each key
#define MAX_VALUES 50

using keytype = int;
using valuetype = long;

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
    int occurrenceCount = 0;
    int occurrences[MAX_OCCURENCES];
} KeyOccurences;

__global__ void fillTable(KeyOccurences *table, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        table[idx].key = -1;
        table[idx].occurrenceCount = 0;
        for(int i = 0;i<MAX_VALUES;i++){
            table[idx].occurrences[i] = -1;
        }
    }   
}

__global__ void fillIndices(int *d_indices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_indices[idx] = idx;
    }   
}

// This functions sorts the keys and rearranges their indices respectively
void sortKeys(keytype *d_keys, int *d_indices, int size) { 
    thrust::device_ptr<keytype> d_keys_ptr(d_keys);
    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + size, d_indices_ptr);
}

// Aggregation kernel based on updated mechanism
__global__ void aggregateIndices(int *keys, int *indices, KeyOccurences *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int key = keys[idx];
        int index = indices[idx];
        
        // Attempt to find or create a bucket for each key
        for (int i = 0; i < size; i++) {
            if (atomicCAS(&output[i].key, -1, key) == -1 || output[i].key == key) {
                // Insert index into the occurrences list for this key
                for (int j = 0; j < MAX_OCCURENCES; j++) {
                    if (atomicCAS(&output[i].occurrences[j], -1, index) == -1) {
                        return;
                    }
                }
            }
        }
    }
}

__global__ void findKeys(keytype* keys, int size, KeyOccurences* table, int tableSize, KeyOccurences* outputTable){ // WTF???
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        for(int i = 0; i<tableSize; i++){
            if (table[i].key == keys[idx]){
                outputTable[idx] = table[i];
            }
        }
    }
}

