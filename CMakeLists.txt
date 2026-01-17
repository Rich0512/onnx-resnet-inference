cmake_minimum_required(VERSION 3.10)
project(OnnxInference)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime
set(ONNX_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime") 
include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)

# OpenCV
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Build executable
add_executable(main main.cpp)

# Link libraries
target_link_libraries(main onnxruntime ${OpenCV_LIBS})

# Copy DLLs to output directory (for Windows)
if(WIN32)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:main>
    )
    
    # Copy OpenCV DLLs
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/opencv/build/x64/vc16/bin/opencv_world4100.dll"
        $<TARGET_FILE_DIR:main>
    )
endif()