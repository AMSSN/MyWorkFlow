cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME "Sample3")
project(${PROJECT_NAME})


set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})

add_executable(${PROJECT_NAME} 
    main.cpp
    LongMat.cpp LongMat.h
)
