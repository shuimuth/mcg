cmake_minimum_required(VERSION 3.12.0)
project(pcg VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

add_executable(pcg main.cu)
target_include_directories(pcg PUBLIC ${CMAKE_SOURCE_DIR}/Common)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
