cmake_minimum_required(VERSION 3.25)
project(demo_Yolov26)
#set(CMAKE_CXX_STANDARD 11)
#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})


# OpenCV
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv)
include_directories(${OPENCV_DIR}/include)
include_directories(${OPENCV_DIR}/include/opencv2)
link_directories(${OPENCV_DIR}/lib/x64/vc15)
if (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    link_libraries(opencv_world452d)
elseif (${CMAKE_BUILD_TYPE} STREQUAL Release)
    link_libraries(opencv_world452)
endif ()

#配置TensorRT路径 (根据实际安装路径修改)
set(TENSORRT_ROOT ${CMAKE_SOURCE_DIR}/../3rdparty/tensorrt)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
link_libraries(nvinfer;nvinfer_dispatch;nvinfer_lean;nvinfer_plugin;nvinfer_vc_plugin;nvonnxparser;nvparsers)

#CUDA
set(CUDA_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/cuda)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)
link_libraries(cudnn;cublas;cudart;nvrtc;cudadevrt;cudart_static)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 创建共享库
add_library(Yolo26Infer SHARED
        src/Yolo26TrtInfer.cpp
        include/Yolo26TrtInfer.h
        )
# 设置C++标准
set_target_properties(Yolo26Infer PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        )

## 为Windows设置导出符号
#if (WIN32)
#    set_target_properties(Yolo26Infer PROPERTIES
#            COMPILE_DEFINITIONS "ONNX_TO_TRT_EXPORTS"
#            )
#endif ()

#导入OnnxToTrt
set(ONNX_TO_TRT_DIR ${CMAKE_SOURCE_DIR}/../demo_OnnxToTrt/cmake-build-debug/OnnxToTrt)
link_directories(${ONNX_TO_TRT_DIR}/lib)
include_directories(${ONNX_TO_TRT_DIR}/include)

## 安装头文件
#install(TARGETS Yolo26Infer
#        EXPORT onnx_to_trtTargets
#        INCLUDES DESTINATION include
#        ARCHIVE DESTINATION lib
#        LIBRARY DESTINATION lib
#        RUNTIME DESTINATION bin
#        )

add_executable(demo_Yolo26
        src/Yolo26TrtInfer.cpp
        include/Yolo26TrtInfer.h
        main.cpp)

target_link_libraries(demo_Yolo26 PRIVATE
        Yolo26Infer
        OnnxToTrt)

# 创建测试可执行文件
add_executable(test_main main.cpp)

# 链接库文件
target_link_libraries(test_main PRIVATE
        Yolo5Infer
        OnnxToTrt)

# 设置输出目录
set_target_properties(demo_Yolo26 test_main PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 添加构建说明
message(STATUS "Building YOLOv5 TensorRT demo with test support")
message(STATUS "Main executable: demo_Yolo26")
message(STATUS "Test executable: test_main")
