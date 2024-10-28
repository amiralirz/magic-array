#pragma once

#include<iostream>

template<typename T>
void peekMemory(T* devMem, int size){
    T* hostMem = (int*)malloc(size * sizeof(T));
    cudaMemcpy(hostMem, devMem, size, cudaMemcpyDeviceToHost);
    for(int i = 0;i<size; i++){
        std::cout<<hostMem[i]<<std::endl;
    }
    free(hostMem);
}