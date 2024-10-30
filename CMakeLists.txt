cmake_minimum_required(VERSION 3.20)

project(indexing LANGUAGES CUDA CXX)

# Add the extended-lambda flag to CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Ensure CUDA architectures are targeted correctly (modify as per your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Example for Turing architecture; adjust based on GPU

# Specify include directories globally for the project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Enable verbose CUDA output (optional, for debugging)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Add executables
# add_executable(indexing src/main.cu)
# add_executable(sort_test src/sort_test.cu)
# add_executable(compress src/compress.cu)
# add_executable(thrust_sort_and_reduce src/thrust_sort_and_reduce.cu)
add_executable(sort_and_filter_test src/sort_and_filter_test.cu)

# Link include directory for each executable
# target_include_directories(indexing PUBLIC include)
# target_include_directories(sort_test PUBLIC include)
# target_include_directories(compress PUBLIC include)
# target_include_directories(thrust_sort_and_reduce PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(sort_and_filter_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Specify CUDA as a required language for each target
# set_target_properties(thrust_sort_and_reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(sort_and_filter_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
