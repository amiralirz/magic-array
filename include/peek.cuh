#pragma once

#include<iostream>
#include"sort_and_filter.cuh"

template<typename T>
void peekMemory(T* devMem, int size){
    T* hostMem = (T*)malloc(size * sizeof(T));
    cudaMemcpy(hostMem, devMem, size * sizeof(T), cudaMemcpyDeviceToHost);
    for(int i = 0;i<size; i++){
        std::cout<<hostMem[i]<<" ";
    }
    std::cout<<std::endl;
    free(hostMem);
}

void peekMemory(KeyOccurences* devMem, int size){
    KeyOccurences* hostMem = (KeyOccurences*)malloc(size * sizeof(KeyOccurences));
    cudaMemcpy(hostMem, devMem, size * sizeof(KeyOccurences), cudaMemcpyDeviceToHost);
    for(int i = 0;i<size; i++){
        std::cout<<hostMem[i].key<<" : ";
        for(int j = 0;hostMem[i].occurrences[j] != -1 && j < MAX_OCCURENCES;j++)
            std::cout<<hostMem[i].occurrences[j]<<" ";
        std::cout<<std::endl;
    }
    free(hostMem);
}

// void peekMemory(TableElement* devMem, int size){
//     TableElement* hostMem = (TableElement*)malloc(size * sizeof(TableElement));
//     cudaMemcpy(hostMem, devMem, size * sizeof(TableElement), cudaMemcpyDeviceToHost);
//     for(int i = 0;i<size; i++){
//         std::cout<<hostMem[i].key<<" : ";
//         for(int j = 0;hostMem[i].values[j] != -1 && j < MAX_OCCURENCES;j++)
//             std::cout<<hostMem[i].values[j]<<" ";
//         std::cout<<std::endl;
//     }
//     free(hostMem);
// }