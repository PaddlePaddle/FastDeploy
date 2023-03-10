PROJECT(paddle2onnx C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

include(cmake/utils.cmake)

ADD_SUBDIRECTORY(paddle2onnx)

option(WITH_STATIC              "Compile Paddle2ONNX with  STATIC"               OFF)
option(PADDLE2ONNX_DEBUG "If open the debug log while converting model"          OFF)
option(MSVC_STATIC_CRT          "Compile Paddle2ONNX with  MSVC STATIC CRT"               ON)
set(ONNX_CUSTOM_PROTOC_PATH "" CACHE STRING "Set yourself Protobufc path")

# Internal flags for convert.h.in
set(WITH_PADDLE2ONNX_STATIC_INTERNAL OFF)  
if(WITH_STATIC)
  set(WITH_PADDLE2ONNX_STATIC_INTERNAL ON CACHE BOOL "" FORCE)
  add_definitions(-DWITH_PADDLE2ONNX_STATIC_INTERNAL_AT_COMPILING)
endif()

configure_file(${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h.in ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h)

if(PADDLE2ONNX_DEBUG)
  add_definitions(-DPADDLE2ONNX_DEBUG)
endif()
# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

# setting max opset version for onnx
# if you build from other version of onnx
# this should be modified
# refer to https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
add_definitions(-DMAX_ONNX_OPSET_VERSION=16)
add_definitions(-DPADDLE2ONNX_LIB)

# Third dependency: onnx
if(NOT TARGET onnx_proto)
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "paddle2onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

  if(ONNX_CUSTOM_PROTOC_PATH)
    if(WIN32)
      if(MSVC_STATIC_CRT)
        # MT
        set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
      else()
        # MD
        set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
      endif()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH};$ENV{PATH}")
    else()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH}:$ENV{PATH}")
    endif()
    set(ENV{PATH} ${ONNX_CUSTOM_PROTOC_PATH})
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third/onnx)
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third/optimizer)
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc ${PROJECT_SOURCE_DIR}/third/optimizer/onnxoptimizer/*.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/third/optimizer/onnxoptimizer/cpp2py_export.cc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" PADDLE2ONNX_VERSION)
string(STRIP "${PADDLE2ONNX_VERSION}" PADDLE2ONNX_VERSION)
if (WITH_STATIC)
    # Here, we use a dummy target (paddle2onnx_dummy) 
    # to form a build dependency tree for paddle2onnx_static lib.
    ADD_LIBRARY(paddle2onnx_dummy STATIC ${ALL_SRCS})
    
    if(APPLE)
        set_target_properties(paddle2onnx_dummy
                              PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")                     
    elseif(MSVC)
        message("------ BUILD WITH MSVC --------")
    else()
        set_target_properties(paddle2onnx_dummy
                              PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")    
    endif()
    target_link_libraries(paddle2onnx_dummy  p2o_paddle_proto onnx)
    # Bundle paddle2onnx static lib here
    bundle_static_library(paddle2onnx_dummy paddle2onnx_static bundle_paddle2onnx)
else ()
    ADD_LIBRARY(paddle2onnx SHARED ${ALL_SRCS})
    if(APPLE)
        set_target_properties(paddle2onnx
                              PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    elseif(MSVC)
        message("------ BUILD WITH MSVC --------")
    else()
        set_target_properties(paddle2onnx
                            PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
        set_target_properties(paddle2onnx
                              PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
        set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS_RELEASE -s)
    endif()
    set_target_properties(paddle2onnx PROPERTIES VERSION ${PADDLE2ONNX_VERSION})
    target_link_libraries(paddle2onnx p2o_paddle_proto onnx)
endif()

#install(
#    DIRECTORY ${PROJECT_SOURCE_DIR}/paddle2onnx
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#    FILES_MATCHING
#    PATTERN "converter.h"
#)
if(WIN32)
  install(
    TARGETS paddle2onnx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
  )
else()
  if(WITH_STATIC)
    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/libpaddle2onnx_static.a
      DESTINATION lib
    )
  else()
    install(
      TARGETS paddle2onnx
      LIBRARY DESTINATION lib
    )
  endif()
endif()
install(
    FILES 
    ${PROJECT_SOURCE_DIR}/paddle2onnx/converter.h
    ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h
    DESTINATION include/paddle2onnx
)


if (BUILD_PADDLE2ONNX_EXE)
  remove_definitions(-DPADDLE2ONNX_LIB)
  ADD_EXECUTABLE(p2o_exec p2o_exec.cpp)
  if (WITH_STATIC)
    TARGET_LINK_LIBRARIES(p2o_exec -Wl,--whole-archive paddle2onnx -Wl,--no-whole-archive)
  else()
    TARGET_LINK_LIBRARIES(p2o_exec paddle2onnx)
  endif()
endif (BUILD_PADDLE2ONNX_EXE)

if(BUILD_PADDLE2ONNX_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  add_library(paddle2onnx_cpp2py_export MODULE ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc ${ALL_SRCS})
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(paddle2onnx_cpp2py_export PRIVATE
                             $<BUILD_INTERFACE:${ONNX_ROOT}>
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  # pybind11 is a header only lib
  find_package(pybind11 2.2)
  if(pybind11_FOUND)
    target_include_directories(paddle2onnx_cpp2py_export PUBLIC
      ${pybind11_INCLUDE_DIRS})
  else()
    if(EXISTS ${PROJECT_SOURCE_DIR}/third/pybind11/include/pybind11/pybind11.h)
      target_include_directories(paddle2onnx_cpp2py_export PUBLIC
      ${PROJECT_SOURCE_DIR}/third/pybind11/include)
    else()
      message(FATAL_ERROR "cannot find pybind")
    endif()
  endif()

  if(APPLE)
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    # whole-archive linker option not available on AIX
    target_sources(paddle2onnx_cpp2py_export
                          PRIVATE $<TARGET_OBJECTS:onnx>)
  else()
    # Assume everything else is like gcc
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx>
                                  "-Wl,--no-whole-archive")
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()

  target_link_libraries(paddle2onnx_cpp2py_export PRIVATE p2o_paddle_proto onnx)

  if(MSVC)
    target_link_libraries(paddle2onnx_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(paddle2onnx_cpp2py_export
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    if(ONNX_USE_PROTOBUF_SHARED_LIBS)
      target_compile_options(onnx_cpp2py_export
                             PRIVATE /wd4251 # 'identifier' : class 'type1' needs to
                                             # have dll-interface to be used by
                                             # clients of class 'type2'
                            )
    endif()
    add_msvc_runtime_flag(paddle2onnx_cpp2py_export)
    add_onnx_global_defines(paddle2onnx_cpp2py_export)
  endif()
endif()
