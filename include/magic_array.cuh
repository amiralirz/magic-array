#include <iostream>
#include <cuda_runtime.h>

class MagicArray {
    private:
        long *dev_keys;           // Stores unique keys on GPU
        int *dev_counts;          // Stores occurrence counts for each key
        int array_size;           // Size of the magic array
        cudaError_t out;          // CUDA error status
        int insertion_pointer;    // Insertion pointer for queue ring

    public:
        MagicArray(int arraySize) : array_size(arraySize), insertion_pointer(0) {
            out = cudaMalloc(&dev_keys, arraySize * sizeof(long));
            if (out != cudaSuccess) {
                std::cout << "Could not allocate GPU memory for keys\n";
            }
            out = cudaMalloc(&dev_counts, arraySize * sizeof(int));
            if (out != cudaSuccess) {
                std::cout << "Could not allocate GPU memory for counts\n";
            }
            cudaMemset(dev_counts, 0, arraySize * sizeof(int)); // Initialize counts to zero
        }

        ~MagicArray() {
            cudaFree(dev_keys);
            cudaFree(dev_counts);
        }

        long* getKeysPointer() {
            return dev_keys;
        }

        int* getCountsPointer() {
            return dev_counts;
        }

        long getAllCounts(){
            long sum = 0;
            for (int i=0; i<array_size; i++) sum += dev_counts[i];
            return sum;
        }

        // Update insertion pointer in a circular manner
        // a device function that ensures circular progression, wrapping the pointer to the start if it reaches the end of magic_keys
        __device__ int getNextInsertionIndex() {
            int current_idx = atomicAdd(&insertion_pointer, 1);
            return current_idx % array_size;
        }
};

// Helper function for binary search in the sorted array
__device__ int binarySearch(long *arr, int left, int right, long key) {
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == key) {
            return mid;
        } else if (arr[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // Key not found
}

__global__ void countOccurrences(long *sorted_keys, int num_keys, long *magic_keys, int *magic_counts, int array_size, int *insertion_pointer) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= array_size) return;

    long key = magic_keys[idx];
    int start_idx = binarySearch(sorted_keys, 0, num_keys - 1, key);

    if (start_idx != -1) {
        // Key found in sorted_keys, count occurrences
        int count = 0;
        for (int i = start_idx; i < num_keys && sorted_keys[i] == key; ++i) {
            count++;
        }
        atomicAdd(&magic_counts[idx], count);
    } else {
        // Key not found, add as new
        int insert_idx = atomicAdd(insertion_pointer, 1) % array_size; // Circular queue ring behavior
        magic_keys[insert_idx] = key;
        atomicAdd(&magic_counts[insert_idx], 1);
    }
}
