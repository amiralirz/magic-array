#include <iostream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdlib.h>
#include <time.h>

const int inputArraySize = 130'000;

int main(){
    srand(0);
    const int N = 130'000;
    long int* host_numbers = (long int*)malloc(N * sizeof(long int));
    for(int i = 0;i<N;i++){
        host_numbers[i] = mrand48();
    }
    
    timespec startSort, endSort;
    
    long int* device_numbers;
    cudaMalloc(&device_numbers, N * sizeof(long int));
    cudaMemcpy(device_numbers, host_numbers, N * sizeof(long int), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &startSort);
    thrust::sort(thrust::device, device_numbers, device_numbers + N);
    clock_gettime(CLOCK_MONOTONIC, &endSort);
    double total_elapsed_time_ms = (endSort.tv_nsec - startSort.tv_nsec) / 1e6; // milisecs = nanosecs / 1e6 
    std::cout<<total_elapsed_time_ms<<" miliseconds" << std::endl;
}