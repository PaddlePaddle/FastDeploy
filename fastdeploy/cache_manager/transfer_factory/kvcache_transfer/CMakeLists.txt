cmake_minimum_required (VERSION 3.5)

project(rdma_comm LANGUAGES CXX)

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Ofast -ffast-math -funroll-loops -march=native -std=c++11")
add_compile_options("-std=c++11")

find_library(IBVERBS_LIBRARY ibverbs)
find_library(RDMACM_LIBRARY rdmacm)

find_package(pybind11 CONFIG REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(rdma_comm MODULE ${PROJECT_SOURCE_DIR}/src/pybind.cpp ${PROJECT_SOURCE_DIR}/src/kvcache_rdma.cpp ${PROJECT_SOURCE_DIR}/src/kvcache_connection.cpp ${PROJECT_SOURCE_DIR}/src/log.cpp)
set_target_properties(rdma_comm PROPERTIES
    OUTPUT_NAME "rdma_comm"
    PREFIX ""
    SUFFIX ".so"
)
target_link_libraries(rdma_comm PRIVATE pybind11::module)

target_link_libraries(rdma_comm LINK_PUBLIC ibverbs pthread)
