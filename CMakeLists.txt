cmake_minimum_required(VERSION 3.20)

project(indexing LANGUAGES CUDA CXX)

# Add the extended-lambda flag to CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_executable(Test src/test.cu) 
target_include_directories(Test PUBLIC include)

add_executable(SortAndReduce src/thrust_sort_and_reduce.cu)
target_include_directories(SortAndReduce PUBLIC include)

add_executable(Indexing src/main.cu)
target_include_directories(Indexing PUBLIC include)
