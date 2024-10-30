#include <iostream>
#include <ctime>
#include "smart_array.cuh"

#define N 100 // Define the number of elements

int main() {
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(0)));

    // Prepare host arrays for keys and values
    keytype h_keys[N];
    valuetype h_values[N];
    for (int i = 0; i < N; i++) {
        h_keys[i] = std::rand() % 10;      // Repeating keys (0-9)
        h_values[i] = static_cast<valuetype>(std::rand());
    }

    // Create a MagicArray instance with a specified table size
    MagicArray magicArray(N);

    // Insert the keys and values into the MagicArray
    magicArray.insert(h_keys, h_values, N);

    // Print the table to verify the output
    std::cout << "Aggregated Key-Index Pairs:\n";
    magicArray.printTable();

    return 0;
}
