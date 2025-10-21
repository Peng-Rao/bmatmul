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

add_custom_command(
    TARGET bmatmul
    POST_BUILD
    COMMAND echo "=== Running NVIDIA Nsight Systems Profiling ==="
    COMMAND nsys profile --stats=true ./bmatmul 256 512 384 16 119
    COMMENT "Profiling CUDA app with Nsight Systems"
)