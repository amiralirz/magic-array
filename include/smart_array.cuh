#pragma once

#include <iostream>
#include "sort_and_filter.cuh"
#include "peek.cuh"

class MagicArray{
    private:
    KeyOccurences *table;   
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

    void insert(keytype* keys, int* indices, int size){
        return;
    }

    void find(keytype* keys, int size){
        KeyOccurences* elements;
        cudaMalloc(&elements, size * sizeof(KeyOccurences));
        findKeys<<<(size+1023)/1024, 1024>>>(keys, size, table, tableSize, elements);
        cudaDeviceSynchronize();
        // peekMemory(elements, size);
        cudaFree(elements);
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