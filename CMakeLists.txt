cmake_minimum_required(VERSION 3.20)

project(indexing LANGUAGES CUDA CXX)

add_executable(indexing src/main.cu) 

target_include_directories(indexing PUBLIC include)