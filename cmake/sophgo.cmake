# get SOPHGO_URL
#set(SOPHGO_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
#set(SOPHGO_VERSION "1.4.0")
#set(SOPHGO_FILE "SOPHGO_runtime-linux-x64-${SOPHGO_VERSION}.tgz")
#set(SOPHGO_URL "${SOPHGO_URL_BASE}${SOPHGO_FILE}")

# download_and_decompress
#download_and_decompress(${SOPHGO_URL} ${CMAKE_CURRENT_BINARY_DIR}/${SOPHGO_FILE} ${THIRD_PARTY_PATH}/install/)

# set path
#set(SOPHGONPU_RUNTIME_PATH ${THIRD_PARTY_PATH}/install/SOPHGO_runtime)

#if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#else ()
#    message(FATAL_ERROR "[SOPHGO.cmake] Only support build SOPHGO in Linux")
#endif ()


#if (EXISTS ${SOPHGONPU_RUNTIME_PATH})
#    set(SOPHGO_RT_LIB ${SOPHGONPU_RUNTIME_PATH}/${SOPHGO_TARGET}/lib/libbmrt.so)
#    include_directories(${SOPHGONPU_RUNTIME_PATH}/${SOPHGO_TARGET}/include)
#else ()
#    message(FATAL_ERROR "[SOPHGO.cmake] download_and_decompress SOPHGO_runtime error")
#endif ()


find_package(libsophon REQUIRED)
message(${LIBSOPHON_LIB_DIRS})
include_directories(${LIBSOPHON_INCLUDE_DIRS})
message(${LIBSOPHON_LIB_DIRS})
set(SOPHGO_RT_LIB ${LIBSOPHON_LIB_DIRS}/libbmrt.so)
#add_executable(${YOUR_TARGET_NAME} ${YOUR_SOURCE_FILES})
#target_link_libraries(${YOUR_TARGET_NAME} ${the_libbmlib.so})