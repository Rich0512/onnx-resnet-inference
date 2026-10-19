cmake_minimum_required(VERSION 3.10)
project(OnnxInference)

set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# GPU Acceleration Option
# ============================================================================
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

# ============================================================================
# Project paths
# ============================================================================
set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}")
set(SRC_DIR "${PROJECT_ROOT}/src")
set(INCLUDE_DIR "${PROJECT_ROOT}/include")
set(THIRD_PARTY_DIR "${PROJECT_ROOT}/third_party")

# ============================================================================
# ONNX Runtime
# ============================================================================
set(ONNX_ROOT "${THIRD_PARTY_DIR}/onnxruntime") 
include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)

# Add CUDA definition and libraries if enabled
if(USE_CUDA)
    message(STATUS "========================================")
    message(STATUS "  CUDA GPU Acceleration: ENABLED")
    
    # Enable CUDA language for .cu compilation with nvcc
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "  Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
    
    add_definitions(-DUSE_CUDA)
    message(STATUS "========================================")
else()
    message(STATUS "========================================")
    message(STATUS "  CUDA GPU Acceleration: DISABLED (CPU only)")
    message(STATUS "  Use -DUSE_CUDA=ON to enable GPU")
    message(STATUS "========================================")
endif()

# ============================================================================
# OpenCV
# ============================================================================
set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================================
# Project headers
# ============================================================================
include_directories(${INCLUDE_DIR})

# ============================================================================
# Python & pybind11
# ============================================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ============================================================================
# Source files
# ============================================================================
set(LIB_SOURCES
    ${SRC_DIR}/preprocessor.cpp
    ${SRC_DIR}/inference.cpp
)

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${LIB_SOURCES}
)

# ============================================================================
# Build executable & Python Module
# ============================================================================
add_executable(main ${SOURCES})

# Create Python extension module
pybind11_add_module(resnet_engine MODULE ${LIB_SOURCES} ${SRC_DIR}/python_bindings.cpp)

# ============================================================================
# TensorRT (Native C++ API)
# ============================================================================
include_directories(${THIRD_PARTY_DIR}/tensorrt/include)
link_directories(${THIRD_PARTY_DIR})

# Link libraries
if(USE_CUDA)
    target_link_libraries(main PRIVATE onnxruntime ${OpenCV_LIBS} CUDA::cudart)
    target_link_libraries(resnet_engine PRIVATE onnxruntime ${OpenCV_LIBS} CUDA::cudart)

    # Native TensorRT 10 Pure C++ Executable (Option B)
    set(TRT_SOURCES
        ${SRC_DIR}/main_trt.cpp
        ${SRC_DIR}/trt_inference.cpp
        ${SRC_DIR}/preprocessor.cpp
        ${SRC_DIR}/cuda_preprocessor.cu
    )
    add_executable(main_trt ${TRT_SOURCES})
    target_link_libraries(main_trt PRIVATE "${THIRD_PARTY_DIR}/nvinfer_10.lib" ${OpenCV_LIBS} CUDA::cudart)
else()
    target_link_libraries(main PRIVATE onnxruntime ${OpenCV_LIBS})
    target_link_libraries(resnet_engine PRIVATE onnxruntime ${OpenCV_LIBS})
endif()

# ============================================================================
# Copy DLLs to output directory (Windows)
# ============================================================================
if(WIN32)
    # Define a helper function to avoid repeating copy commands
    function(copy_dlls_to_target target_name)
        # Core ONNX Runtime DLL
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
        
        # OpenCV DLL 
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${THIRD_PARTY_DIR}/opencv/build/x64/vc16/bin/opencv_world4100.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
        
        # CUDA DLLs (only if CUDA enabled)
        if(USE_CUDA)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNX_ROOT}/lib/onnxruntime_providers_shared.dll"
                $<TARGET_FILE_DIR:${target_name}>
            )
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNX_ROOT}/lib/onnxruntime_providers_cuda.dll"
                $<TARGET_FILE_DIR:${target_name}>
            )
        endif()
    endfunction()

    copy_dlls_to_target(main)
    copy_dlls_to_target(resnet_engine)
    if(USE_CUDA)
        copy_dlls_to_target(main_trt)
    endif()
endif()