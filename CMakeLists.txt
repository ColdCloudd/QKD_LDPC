cmake_minimum_required(VERSION 3.30.0)
project(QKD_LDPC VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
if(NOT(EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
    NAME BS_thread_pool
    GITHUB_REPOSITORY bshoshany/thread-pool
    VERSION 4.1.0)
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${BS_thread_pool_SOURCE_DIR}/include)

CPMAddPackage(
    NAME fmt
    GIT_TAG 11.0.2
    GITHUB_REPOSITORY fmtlib/fmt)

CPMAddPackage(
    NAME indicators
    GITHUB_REPOSITORY p-ranav/indicators
    VERSION 2.3)

CPMAddPackage(
    NAME xoshiro_cpp
    GITHUB_REPOSITORY Reputeless/Xoshiro-cpp
    VERSION 1.1)
include_directories(${xoshiro_cpp_SOURCE_DIR})

CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.11.3
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS
        "JSON_BuildTests OFF")

add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_executable(QKD_LDPC 
    src/main.cpp
    src/config.hpp
    src/config.cpp
    src/utils.hpp
    src/utils.cpp
    src/array_and_matrix_operations.hpp
    src/array_and_matrix_operations.cpp
    src/qkd_ldpc_algorithm.hpp
    src/qkd_ldpc_algorithm.cpp
    src/simulation.hpp
    src/simulation.cpp
    )
target_include_directories(QKD_LDPC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(QKD_LDPC BS_thread_pool fmt indicators nlohmann_json)

option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_executable(QKD_LDPC_Example 
    example/qkd_ldpc_example.cpp
    src/config.hpp
    src/config.cpp
    src/utils.hpp
    src/utils.cpp
    src/array_and_matrix_operations.hpp
    src/array_and_matrix_operations.cpp
    src/qkd_ldpc_algorithm.hpp
    src/qkd_ldpc_algorithm.cpp
    )
    target_include_directories(QKD_LDPC_Example PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(QKD_LDPC_Example fmt nlohmann_json)
endif()