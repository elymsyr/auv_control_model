cmake_minimum_required(VERSION 3.10)
project(cpp_gpu_model_app)

# Set correct LibTorch path
set(LIBTORCH_PATH "libtorch")  # Path to your extracted LibTorch
set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")

# Find Torch package
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include ${TORCH_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp
    src/model_inference.cpp
    src/utils.cpp
)

add_executable(cpp_gpu_model_app ${SOURCES})

# Only link with TORCH_LIBRARIES - no need for explicit CUDA libs
target_link_libraries(cpp_gpu_model_app 
    PRIVATE 
    ${TORCH_LIBRARIES}
)