cmake_minimum_required(VERSION 3.25)
set(PROJECT_NAME "Sample2_thread")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)


# 显式列出源文件和头文件
set(SOURCES
        "src/StdThreadExample.cpp"
        "src/AsyncExample.cpp"
        )

set(HEADERS
        "src/StdThreadExample.h"
        "src/AsyncExample.h"
        )

# 创建可执行文件（必须在 target_* 命令之前）
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} main.cpp)


# 使用 target_include_directories 替代全局 include_directories
# 确保能包含当前目录的头文件
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# 链接找到的库
# target_link_libraries(Sample2_thread PRIVATE ${LMTRT_LIBRARY})
