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
# Source files
# ============================================================================
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/preprocessor.cpp
    ${SRC_DIR}/inference.cpp
)

# ============================================================================
# Build executable
# ============================================================================
add_executable(main ${SOURCES})

# Link libraries
if(USE_CUDA)
    target_link_libraries(main onnxruntime ${OpenCV_LIBS} CUDA::cudart)
else()
    target_link_libraries(main onnxruntime ${OpenCV_LIBS})
endif()

# ============================================================================
# Copy DLLs to output directory (Windows)
# ============================================================================
if(WIN32)
    # Core ONNX Runtime DLL
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:main>
    )
    
    # OpenCV DLL
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${THIRD_PARTY_DIR}/opencv/build/x64/vc16/bin/opencv_world4100.dll"
        $<TARGET_FILE_DIR:main>
    )
    
    # CUDA DLLs (only if CUDA enabled)
    if(USE_CUDA)
        add_custom_command(TARGET main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:main>
        )
        add_custom_command(TARGET main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:main>
        )
    endif()
endif()