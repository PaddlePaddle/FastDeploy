cmake_minimum_required(VERSION 3.21)
project(poros)
set(CMAKE_CXX_STANDARD 14)

option(BUILD_STATIC "build lib${PROJECT_NAME}.a static lib" OFF)
option(BUILD_KERNEL "build lib${PROJECT_NAME}-kernel.so shared lib" OFF)
option(BUILD_STATIC_KERNEL "build lib${PROJECT_NAME}-kernel.a static lib" OFF)
option(BUILD_TOOL "build ${PROJECT_NAME}-tool, an executable binary output" OFF)
option(TEST "build for test. copy '.so' to site-packages automatically after compile" OFF)
option(DEBUG "build for debug. add '-g' flag to gcc for detailed debug information" ON)
option(UT "build for unit test" OFF)
option(ABI "build ${PROJECT_NAME} with application binary interface (ABI) is on" OFF)

# abi configuration
if (NOT ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
else ()
    if (BUILD_TOOL OR UT)
        message(FATAL_ERROR "${PROJECT_NAME}-tool or unit test are not supported when abi is on.")
    endif()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
endif ()

# minimum requirements
set(PYTHON_MINIMUM_VERSION 3.6)
set(CUDA_MINIMUM_VERSION 10.2)
set(PYTORCH_MINIMUM_VERSION 1.9)
set(TENSORRT_MINIMUM_VERSION 8.2)
set(CUDNN_MINIMUM_VERSION 8.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find cuda
find_package(CUDAToolkit ${CUDA_MINIMUM_VERSION} REQUIRED)

# find python3
find_package(Python3 ${PYTHON_MINIMUM_VERSION} REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

if (NOT Python3_SITELIB)
    message(FATAL_ERROR "site-packages not found. ")
else ()
    message(STATUS "site-packages: ${Python3_SITELIB}")
endif ()

# find pytorch
find_package(Torch ${PYTORCH_MINIMUM_VERSION} REQUIRED HINTS ${Python3_SITELIB})

# find TensorRT
find_package(TensorRT ${TENSORRT_MINIMUM_VERSION} REQUIRED)
get_filename_component(TENSORRT_LIB_DIR ${TensorRT_LIBRARIES} DIRECTORY)


# find CUDNN
find_package(CUDNN ${CUDNN_MINIMUM_VERSION} REQUIRED)


## release headers
# engine
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/engine/iengine.h" "${PROJECT_SOURCE_DIR}/poros/engine/engine_context.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/engine")
# compile
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/compile/poros_module.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/compile")
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/compile/compile.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/compile")
# converter
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/converter/iconverter.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/converter")
# iplugin
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/iplugin/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/iplugin")
## context
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/context/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/context")
## context
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/context/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/context")
## lowering
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/lowering/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/lowering")
## util
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/util/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/util")
## log
file(GLOB headers "${PROJECT_SOURCE_DIR}/poros/log/*.h")
file(COPY ${headers} DESTINATION "${PROJECT_SOURCE_DIR}/build/include/poros/log")


include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_PATH})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(poros/compile)


add_compile_options(-D__const__= -D_GNU_SOURCE)
add_compile_options(-lpthread -lcrypto -lrt -ldl -lz -fPIC -rdynamic)
add_compile_options(-std=c++17 -O2 -g -pipe -W -Wall -fPIC -Wno-deprecated-declarations -Wno-unused-parameter)
if (DEBUG)
    add_compile_options(-g) # for debug
endif ()

add_compile_options(
        -Wall
        -Wno-comment
        -Wno-error=implicit-fallthrough
        -Wno-error=unused-but-set-variable
        -Wno-error=misleading-indentation
        -Wno-error=unused-function
        -Wno-error=terminate
        -Wno-unused-parameter
        -Wno-deprecated-declarations
)


file(
        GLOB POROS_CPP_FILES
        "./poros/*/*.cpp"
        "./poros/converter/*/*.cpp"
        "./poros/converter/gpu/plugins/*.cpp"
)


# libporos.so
add_library(${PROJECT_NAME} SHARED ${POROS_CPP_FILES})

#set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/poros/lib)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)

#add_custom_command(
#    TARGET ${PROJECT_NAME}
#    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/output/lib/lib${PROJECT_NAME}.so ${CMAKE_CURRENT_SOURCE_DIR}/python/poros/lib/
#)

# copy libporos.so to python site-packages/poros/lib for testing
if (TEST)
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMENT "copy ${LIBRARY_OUTPUT_PATH}/lib${PROJECT_NAME}.so to ${Python3_SITELIB}/poros/lib/lib${PROJECT_NAME}.so for rapid testing"
            COMMAND ${CMAKE_COMMAND} -E copy ${LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}.so ${Python3_SITELIB}/poros/lib/lib${PROJECT_NAME}.so
    )
endif ()


if (BUILD_STATIC)
    add_library(${PROJECT_NAME}-static STATIC ${POROS_CPP_FILES})
    set_target_properties(${PROJECT_NAME}-static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif ()

# build gflags
set(GFLAGS_NAMESPACE google)
add_subdirectory(third_party/gflags)


find_package(BLAS)

add_custom_target(
        Creating_Symlink ALL
        COMMAND_EXPAND_LISTS
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${TENSORRT_LIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrtlib
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Creating Symlink ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrtlib -> ${TENSORRT_LIB_DIR} "
        VERBATIM
)

# executable
if (BUILD_TOOL)
    set(POROS_TOOL ${PROJECT_NAME}-tool)


    add_executable(${POROS_TOOL})

    target_sources(${POROS_TOOL} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/main.cpp)
    target_sources(${POROS_TOOL} PUBLIC ${POROS_CPP_FILES})

    target_link_libraries(${POROS_TOOL} gflags::gflags)
    target_link_libraries(${POROS_TOOL} TensorRT::TensorRT TensorRT::Plugin)
    target_link_libraries(${POROS_TOOL} torch)
#    target_link_libraries(${POROS_TOOL} CUDA::toolkit)
    target_link_libraries(${POROS_TOOL} CUDA::cudart CUDA::cusolver CUDA::cublas CUDA::cusolver CUDA::cusparse)
    target_link_libraries(${POROS_TOOL} BLAS::BLAS)

endif ()



# kernel
file(
        GLOB POROS_KERNEL_CPP_FILES
        ./poros/compile/*.cpp
        ./poros/context/*.cpp
        ./poros/iplugin/*.cpp
        ./poros/log/*.cpp
        ./poros/lowering/*.cpp
        ./poros/util/*.cpp
        ./poros/engine/engine.cpp
)

# kernel SHARED
if (BUILD_KERNEL)
    add_library(${PROJECT_NAME}-kernel SHARED ${POROS_KERNEL_CPP_FILES})
endif ()

# kernel STATIC
if (BUILD_STATIC_KERNEL)
    add_library(${PROJECT_NAME}-kernel-static STATIC ${POROS_KERNEL_CPP_FILES})
    set_target_properties(${PROJECT_NAME}-kernel-static PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-kernel)
endif ()

if (UT)
    add_subdirectory(third_party/googletest)
    add_subdirectory(unittest)
endif ()
