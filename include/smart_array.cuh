#pragma once

#include <iostream>
#include <cuda.h>
#include "sort_and_filter.cuh"
#include "peek.cuh"

class MagicArray{
    private:
    TableElement *table;   
    int tableSize;     
    cudaError_t out;

    public:
    MagicArray(int arraySize){
        out = cudaMalloc(&table ,arraySize * sizeof(KeyOccurences));
        if (out != cudaSuccess){
            std::cout<<"Could not allocate GPU memory\n";
            return;
        }
        tableSize = arraySize;
        fillTable<<<(arraySize + 1023)/1024, 1024>>>(table, tableSize);
        cudaDeviceSynchronize();
    }
    
    ~MagicArray(){
        cudaFree(table);
    }

    // first sort the keys, then reduce it (using sort_and_filter.cuh methods)
    void insert(keytype* h_keys, valuetype* h_values, int size) {
        // Step 1: Allocate and initialize device arrays
        keytype *d_keys;
        int *d_indices;
        cudaMalloc(&d_keys, size * sizeof(keytype));
        cudaMalloc(&d_indices, size * sizeof(int));

        // Copy host keys to device
        cudaMemcpy(d_keys, h_keys, size * sizeof(keytype), cudaMemcpyHostToDevice);
        fillIndices<<<(size + 255) / 256, 256>>>(d_indices, size); // Fill indices

        // Step 2: Sort the keys and indices
        sortKeys(d_keys, d_indices, size);

        // Step 3: Aggregate the sorted keys and indices into the table
        aggregateIndices<<<(size + 255) / 256, 256>>>(d_keys, d_indices, table, size);
        cudaDeviceSynchronize();

        // Clean up
        cudaFree(d_keys);
        cudaFree(d_indices);
    }


    void find(keytype* keys, int size){
        return;
    }

    void erase(keytype* keys){
        return;
    }

    void printTable(){
        KeyOccurences* h_table = (KeyOccurences*)malloc(tableSize * sizeof(KeyOccurences));
        cudaMemcpy(h_table, table, tableSize * sizeof(KeyOccurences), cudaMemcpyDeviceToHost);
        for(int i = 0;i<tableSize;i++){
            if(h_table[i].key != -1){
                std::cout<<h_table[i].key<<" : ";
                for(int j = 0;j<MAX_OCCURENCES;j++){
                    if(h_table[i].occurrences[j] != -1){
                        std::cout<<h_table[i].occurrences[j]<<" ";
                    }
                }
                std::cout<<std::endl;
            }
        }
        free(h_table);
    }
};