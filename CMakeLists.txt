cmake_minimum_required(VERSION 3.20)

project(indexing LANGUAGES CUDA CXX)

# Add the extended-lambda flag to CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_executable(indexing src/main.cu) 
add_executable(sort_test src/sort_test.cu) 
# add_executable(compress src/compress.cu) 
add_executable(thrust_sort_and_reduce src/thrust_sort_and_reduce.cu)

target_include_directories(indexing PUBLIC include)
target_include_directories(sort_test PUBLIC include)
# target_include_directories(compress PUBLIC include)
target_include_directories(thrust_sort_and_reduce PUBLIC include)
