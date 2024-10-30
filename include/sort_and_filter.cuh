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

typedef struct{
    keytype key;
    int valueCount = 0;
    valuetype values[MAX_VALUES];
} TableElement;

// this function copies the keys and their indices and writes them in seperate arrays
__global__ void extractKeys(KeyValuePair *input, int *keys, int *indices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keys[idx] = input[idx].key;
        indices[idx] = idx;
    }
}


__global__ void fillTable(TableElement *table, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        table[idx].key = -1;
        table[idx].valueCount = -1;
        for(int i = 0;i<MAX_VALUES;i++){
            table[idx].values[i] = -1;
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

__global__ void reduce(keytype *keys, int *indices, KeyOccurences *table, int size, int tableSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keytype key = keys[idx];
        int index = indices[idx];

        // inserting to an existing bucekt
        for (int i = 0; i < tableSize; i++) {
            if (table[i].key == key) {
                for (int j = 0; j < MAX_OCCURENCES; j++) {
                    if (table[i].occurrences[j] == -1) {
                        if(atomicCAS(&table[i].occurrences[j], -1, index) == -1)
                            return;
                    }
                }
            }
        }

        // inserting to a new bucket if no exisiting buckets match
        for (int i = 0; i < tableSize; i++) {
            if (table[i].key == -1) {
                int old = atomicCAS(&table[i].key, -1, key);
                if(old == -1 || old == key){ // if the old value is the same as key, we can insert too
                    for (int j = 0; j < MAX_OCCURENCES; j++) {
                        if (table[i].occurrences[j] == -1) {
                            table[i].occurrences[j] = index;
                            if(atomicCAS(&table[i].occurrences[j], -1, index) == -1)
                                return;
                        }
                    }
                }
            }
        }
    }
}
