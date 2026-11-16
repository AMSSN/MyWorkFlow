cmake_minimum_required(VERSION 3.16)
project(Samples)

# 公共编译选项（可选）
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
    add_compile_options("/source-charset:utf-8")
endif()

set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# ========== 公共第三方库配置 ==========
# 统一查找和配置，所有子项目共享

# OpenCV
set(OPENCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv)
include_directories(${OPENCV_DIR}/include)
include_directories(${OPENCV_DIR}/include/opencv2)
link_directories(${OPENCV_DIR}/lib/x64/vc15)

if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    link_libraries(opencv_world452d)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    link_libraries(opencv_world452)
endif()

# Halcon
set(HALCON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/halcon)
include_directories(${HALCON_DIR}/include)
include_directories(${HALCON_DIR}/include/halconcpp)
link_directories(${HALCON_DIR}/lib/x64-win64)
link_libraries(halconcpp)

#Tensorrt
set(TENSORRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/tensorrt)
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
link_libraries(nvinfer;nvinfer_dispatch;nvinfer_lean;nvinfer_plugin;nvinfer_vc_plugin;nvonnxparser;nvparsers)

#CUDA
set(CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cuda)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)
link_libraries(cudnn;cublas;cudart;nvrtc;cudadevrt;cudart_static)

# Openssl
set(OPENSSL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/OpenSSL)
include_directories(${OPENSSL_DIR}/include)
link_directories(${OPENSSL_DIR}/lib/x64/MD)
link_libraries(libcrypto)



# ========== 添加子项目 ==========
#add_subdirectory(Sample1)
#add_subdirectory(Sample2_thread)
add_subdirectory(Sample3)