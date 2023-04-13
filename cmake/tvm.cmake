# set path
set(TVM_BASE_PATH "/Users/zhengbicheng/Code/Hackathon/AddTVMForFastDeploy/FastDeploy/tvm")
set(TVM_RUNTIME_PATH "${THIRD_PARTY_PATH}/install/tvm")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TVM_RUNTIME_PATH}")

# include lib
if (EXISTS ${TVM_RUNTIME_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TVM_RUNTIME_PATH}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${TVM_BASE_PATH}" "${TVM_RUNTIME_PATH}")
    set(TVM_RUNTIME_LIB ${TVM_RUNTIME_PATH}/lib/libtvm_runtime.dylib)
    include_directories(${TVM_RUNTIME_PATH}/include)
    include(${TVM_RUNTIME_PATH}/lib/cmake/tvm/tvmConfig.cmake)
    add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
else ()
    message(FATAL_ERROR "[tvm.cmake] TVM_RUNTIME_PATH does not exist.")
endif ()

