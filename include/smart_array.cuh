#include<iostream>

__global__ void insert(){
    int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
}

class MagicArray{
    private:
    long * dev_memory;        
    cudaError_t out;

    public:
    MagicArray(int arraySize){
        out = cudaMalloc(&dev_memory ,arraySize * sizeof(long));
        if (out != cudaSuccess){
            std::cout<<"Could not allocate GPU memory\n";
        }
    }
    
    ~MagicArray(){
        cudaFree(dev_memory);
    }

    cudaError_t Insert(long* dataPointer, int n){

    }

};