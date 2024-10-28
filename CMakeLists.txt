cmake_minimum_required(VERSION 3.20)

project(Indexing LANGUAGES CUDA CXX)

add_executable(Inedxing src/main.cu) 
add_executable(Test src/test.cu) 
target_include_directories(Inedxing PUBLIC include)
target_include_directories(Test PUBLIC include)