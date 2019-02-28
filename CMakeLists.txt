cmake_minimum_required(VERSION 3.5.1)

project(p3)

find_package(CUDA REQUIRED)

include_directories(
src
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CUDA_NVCC_FLAGS -O3;-G;-g)


cuda_add_executable(
	p3
	src/input_image.cuh
	src/Source.cu
)







