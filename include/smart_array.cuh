#include<iostream>

class MagicArray{
    private:
    long * dev_memory;        
    cudaError_t out;

    public:
    MagicArray();
    MagicArray(int arraySize){
        out = cudaMalloc(&dev_memory ,arraySize * sizeof(long));
        if (out != cudaSuccess){
            std::cout<<"Could not allocate GPU memory\n";
        }
    }
    

};