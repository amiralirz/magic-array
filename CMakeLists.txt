cmake_minimum_required(VERSION 3.20)

project(Indexing LANGUAGES CUDA CXX)

add_executable(Indexing src/main.cu) 

target_include_directories(Indexing PUBLIC include)