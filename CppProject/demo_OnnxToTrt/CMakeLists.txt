cmake_minimum_required(VERSION 3.25)
project(demo_OnnxToTrt)

set(CMAKE_CXX_STANDARD 11)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

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

# 创建DLL
add_library(OnnxToTrt SHARED
        src/OnnxToTrt.cpp
        include/OnnxToTrt.h
        )
# 设置C++标准
set_target_properties(OnnxToTrt PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        )

## 链接TensorRT核心库
#target_link_libraries(OnnxToTrt
#        nvinfer
#        nvinfer_plugin
#        onnx
#        onnx_proto
#        )

## 为Windows设置导出符号
#if (WIN32)
#    set_target_properties(OnnxToTrt PROPERTIES
#            COMPILE_DEFINITIONS "ONNX_TO_TRT_EXPORTS"
#            )
#endif ()

# 安装头文件
install(TARGETS OnnxToTrt
        EXPORT onnx_to_trtTargets
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )

# 创建测试可执行文件
add_executable(demo_OnnxToTrt
        main.cpp)
# 链接OnnxToTrt DLL
target_link_libraries(demo_OnnxToTrt OnnxToTrt)