###
# Set minimum version of CMake. Since command 'project' use
# VERSION sub-option we need at least 3.0.
# Note: If you use 2.6 or 2.4, God kills a kitten. Seriously.
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

####
# Set variables:
#   * PROJECT_NAME
#   * PROJECT_VERSION
project(dlpack VERSION 0.6 LANGUAGES C CXX)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

####
# Setup the compiler options

# set c++ standard to c++11.
# Note: not working on CMake 2.8. We assume that user has
#       a compiler with C++11 support.

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

option(BUILD_DOCS "Set to ON to build documentation" OFF)
option(BUILD_MOCK "Build mock executable" ON)

if(BUILD_DOCS)
    add_subdirectory(docs)
endif(BUILD_DOCS)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(BUILD_MOCK)
  set(DLPACK_MOCK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/contrib/mock_main.cc
                      ${CMAKE_CURRENT_SOURCE_DIR}/contrib/mock_c.c)
  add_executable(mock ${DLPACK_MOCK_SRCS})
endif()

add_library(dlpack INTERFACE)
add_library(${PROJECT_NAME}::dlpack ALIAS dlpack)

target_include_directories(
    dlpack
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/contrib>
)

if(BUILD_MOCK)
  target_link_libraries(mock PRIVATE dlpack)
endif()

# Installation (https://github.com/forexample/package-example) {

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

include(CMakePackageConfigHelpers)

# Use:
# * PROJECT_VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Use:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
    "cmake/template/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

install(
    TARGETS dlpack
    EXPORT "${TARGETS_EXPORT_NAME}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)

# }
