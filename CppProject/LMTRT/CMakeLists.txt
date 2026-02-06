cmake_minimum_required(VERSION 3.25)
project(LMTRT)

set(CMAKE_CXX_STANDARD 11)
# 👇 添加这一行：让 Windows 自动导出所有符号
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(LMTRT SHARED LMTRT.cpp)


cmake_minimum_required(VERSION 3.10)
project(OnnxToTrt)

# 配置TensorRT路径 (根据实际安装路径修改)
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")

# 设置包含路径和库路径
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# 创建DLL
add_library(onnx_to_trt SHARED
        src/onnx_to_trt.cpp
        include/onnx_to_trt.h
        )

# 设置C++标准
set_target_properties(onnx_to_trt PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        )

# 链接TensorRT核心库
target_link_libraries(onnx_to_trt
        nvinfer
        nvinfer_plugin
        onnx
        onnx_proto
        )

# 为Windows设置导出符号
if(WIN32)
    set_target_properties(onnx_to_trt PROPERTIES
            COMPILE_DEFINITIONS "ONNX_TO_TRT_EXPORTS"
            )
endif()

# 安装头文件
install(TARGETS onnx_to_trt
        EXPORT onnx_to_trtTargets
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )