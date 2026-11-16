cmake_minimum_required(VERSION 3.25)
project(Sample1)

set(CMAKE_CXX_STANDARD 11)

# 创建可执行文件（必须在 target_* 命令之前）
add_executable(Sample1 main.cpp)

# 使用 target_include_directories 替代全局 include_directories
target_include_directories(Sample1 PRIVATE ${CMAKE_SOURCE_DIR})

# 查找 LMTRT 的导入库 LMTRT.lib
find_library(LMTRT_LIBRARY
        NAMES LMTRT
        PATHS "${CMAKE_SOURCE_DIR}"          # 示例路径：项目下的 lib 文件夹
        NO_DEFAULT_PATH
        )

# 链接找到的库
target_link_libraries(Sample1 PRIVATE ${LMTRT_LIBRARY})
