cmake_minimum_required(VERSION 3.18)
project(bmatmul LANGUAGES CXX CUDA)

# RTX 30 Ampere
set(CMAKE_CUDA_ARCHITECTURES 86)
# RTX 40 Ada
# set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(bmatmul bmatmul.cu)

set_target_properties(bmatmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)
