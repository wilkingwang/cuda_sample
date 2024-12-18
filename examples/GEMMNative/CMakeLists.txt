cmake_minimum_required(VERSION 3.9)

enable_language(CUDA)

find_package(CUDAToolkit 11.6 REQUIRED)

set(LIB_DEPENDENCY cublas Core Framework)

include_directories(${CUDA_INCLUDE_DIRS})
add_example(GEMMNative "" LIB_DEPENDENCY)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
target_link_directories(${PROJECT_NAME} PUBLIC include ${CUDA_INCLUDE_DIRS})