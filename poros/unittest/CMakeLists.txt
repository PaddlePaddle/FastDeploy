cmake_minimum_required(VERSION 3.21)
project(unittest)
set(CMAKE_CXX_STANDARD 14)

enable_testing()
include(GoogleTest)

set(GRAPHTEST "graph_test"  )

file(
        GLOB UT_FILES
        "./op_fuser/*.cpp"
        "../poros/lowering/fuse_*.cpp"
)
list(APPEND UT_FILES
        "../poros/lowering/op_fuse_pass.cpp"
        "../poros/util/graph_test_helper.cpp")

add_executable(${GRAPHTEST} ${UT_FILES})
target_link_libraries(${GRAPHTEST} gtest_main)
target_link_libraries(${GRAPHTEST} gflags::gflags)
#target_link_libraries(${UNITTEST} TensorRT::TensorRT)
target_link_libraries(${GRAPHTEST} torch)
#target_link_libraries(${UNITTEST} CUDA::cudart CUDA::cusolver CUDA::cublas CUDA::cusolver CUDA::cusparse)

# unit test
set(UNITTEST "unit_test"  )

file(
        GLOB UT_FILES
        "../poros/*/*.cpp"
        "../poros/converter/*/*.cpp"
        "./converter/*.cpp"
)

add_executable(${UNITTEST} ${UT_FILES})
target_link_libraries(${UNITTEST} gtest_main)
target_link_libraries(${UNITTEST} gflags::gflags)
target_link_libraries(${UNITTEST} TensorRT::TensorRT TensorRT::Plugin)
target_link_libraries(${UNITTEST} torch)
target_link_libraries(${UNITTEST} CUDA::cudart CUDA::cusolver CUDA::cublas CUDA::cusolver CUDA::cusparse)
