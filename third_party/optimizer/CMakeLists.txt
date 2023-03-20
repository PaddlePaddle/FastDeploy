cmake_minimum_required(VERSION 3.1)

project(onnx_optimizer C CXX)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(protobuf_BUILD_TESTS "" OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/protobuf/cmake)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR}/third_party/onnx)
add_subdirectory(${ONNX_ROOT})

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" ONNX_OPTIMIZER_VERSION)
string(STRIP "${ONNX_OPTIMIZER_VERSION}" ONNX_OPTIMIZER_VERSION)

file(GLOB_RECURSE onnx_opt_srcs "onnxoptimizer/*.cc"
    "onnxoptimizer/*.h"
    )
list(REMOVE_ITEM onnx_opt_srcs "${PROJECT_SOURCE_DIR}/onnxoptimizer/cpp2py_export.cc")

add_library(onnx_optimizer ${onnx_opt_srcs})
target_link_libraries(onnx_optimizer PUBLIC onnx)
target_include_directories(onnx_optimizer PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

add_executable(onnx_optimizer_exec examples/onnx_optimizer_exec.cpp)
target_link_libraries(onnx_optimizer_exec onnx_optimizer)

if(BUILD_ONNX_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  add_library(onnx_opt_cpp2py_export MODULE "onnxoptimizer/cpp2py_export.cc")
  set_target_properties(onnx_opt_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(onnx_opt_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(onnx_opt_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(onnx_opt_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(onnx_opt_cpp2py_export PRIVATE
                             $<BUILD_INTERFACE:${ONNX_ROOT}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})
  # pybind11 is a header only lib
  find_package(pybind11 2.2)
  if(pybind11_FOUND)
    target_include_directories(onnx_opt_cpp2py_export PUBLIC
      ${pybind11_INCLUDE_DIRS})
  else()
    if(EXISTS ${ONNX_ROOT}/third_party/pybind11/include/pybind11/pybind11.h)
      target_include_directories(onnx_opt_cpp2py_export PUBLIC
        ${ONNX_ROOT}/third_party/pybind11/include)
    else()
      message(FATAL_ERROR "cannot find pybind")
    endif()
  endif()

  if(APPLE)
    set_target_properties(onnx_opt_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    target_link_libraries(onnx_opt_cpp2py_export
                          PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx_optimizer>)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default
    target_link_libraries(onnx_opt_cpp2py_export
                          PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx_optimizer>)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    # whole-archive linker option not available on AIX
    target_sources(onnx_opt_cpp2py_export
                          PRIVATE $<TARGET_OBJECTS:onnx_optimizer>)
  else()
    # Assume everything else is like gcc
    target_link_libraries(onnx_opt_cpp2py_export
                          PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx_optimizer>
                                  "-Wl,--no-whole-archive")
    set_target_properties(onnx_opt_cpp2py_export
                          PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()

  target_link_libraries(onnx_opt_cpp2py_export PRIVATE onnx_optimizer)

  if(MSVC)
    find_package(PythonInterp ${PY_VERSION} REQUIRED)
    find_package(PythonLibs ${PY_VERSION} REQUIRED)
    target_link_libraries(onnx_opt_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(onnx_opt_cpp2py_export
                           PRIVATE /MP
                                   /WX
                                   /wd4800 # disable warning type' : forcing
                                           # value to bool 'true' or 'false'
                                           # (performance warning)
                                   /wd4503 # identifier' : decorated name length
                                           # exceeded, name was truncated
                                   /wd4146 # unary minus operator applied to
                                           # unsigned type, result still
                                           # unsigned from include\google\protob
                                           # uf\wire_format_lite.h
                                 /wd4244 # 'argument': conversion from 'google::
                                         # protobuf::uint64' to 'int', possible
                                         # loss of data
                                 /wd4267 # Conversion from 'size_t' to 'int',
                                         # possible loss of data
                                 /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    if(ONNX_USE_PROTOBUF_SHARED_LIBS)
      target_compile_options(onnx_opt_cpp2py_export
                             PRIVATE /wd4251 # 'identifier' : class 'type1' needs to
                                             # have dll-interface to be used by
                                             # clients of class 'type2'
                            )
    endif()
    add_msvc_runtime_flag(onnx_opt_cpp2py_export)
  endif()
endif()

include(GNUInstallDirs)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/onnxoptimizer
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h")

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXOptimizerConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfigVersion.cmake
  @ONLY)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXOptimizerConfig.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfig.cmake
  @ONLY)
install(FILES
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ONNXOptimizer
  COMPONENT dev)
install(EXPORT ONNXOptimizerTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ONNXOptimizer")
install(TARGETS
  onnx_optimizer
  EXPORT ONNXOptimizerTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})

