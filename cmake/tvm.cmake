# set path

set(TVM_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(TVM_VERSION "0.12.0")
set(TVM_SYSTEM "")

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if (${CMAKE_SYSTEM} MATCHES "Darwin")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
        set(TVM_SYSTEM "macos-arm64")
    endif ()
else ()
    error("TVM only support MacOS in Arm64")
endif ()
set(TVM_FILE "tvm-${TVM_SYSTEM}-${TVM_VERSION}")
set(TVM_URL "${TVM_URL_BASE}${TVM_FILE}.tgz")


download_and_decompress(${TVM_URL} ${CMAKE_CURRENT_BINARY_DIR}/tvm ${THIRD_PARTY_PATH}/install/)
set(TVM_RUNTIME_PATH "${THIRD_PARTY_PATH}/install/tvm")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TVM_RUNTIME_PATH}")

# include lib
if (EXISTS ${TVM_RUNTIME_PATH})
    if (${CMAKE_SYSTEM} MATCHES "Darwin")
        set(TVM_RUNTIME_LIB ${TVM_RUNTIME_PATH}/lib/libtvm_runtime.dylib)
    endif ()
    include_directories(${TVM_RUNTIME_PATH}/include)
    include(${TVM_RUNTIME_PATH}/lib/cmake/tvm/tvmConfig.cmake)
    add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
else ()
    error(FATAL_ERROR "[tvm.cmake] TVM_RUNTIME_PATH does not exist.")
endif ()