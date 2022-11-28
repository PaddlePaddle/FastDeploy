#####################################
## tensorrt specific configuration ##
#####################################

set(_TensorRT_SEARCHES)

if (DEFINED ENV{TENSORRT_ROOT})
    set(_TensorRT_SEARCH_ROOT PATHS $ENV{TENSORRT_ROOT} NO_DEFAULT_PATH)
    list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif ()

if (DEFINED ENV{TensorRT_INCLUDE_DIR})
    set(TensorRT_INCLUDE_DIR $ENV{TensorRT_INCLUDE_DIR})
endif ()

if (DEFINED ENV{TensorRT_LIBRARY})
    set(TensorRT_LIBRARY $ENV{TensorRT_LIBRARY})
endif ()

# appends some common paths
set(_TensorRT_SEARCH_NORMAL
        PATHS "/usr/src/tensorrt/" # or custom tensorrt path
        PATHS "/usr/local/tensorrt/" # or custom tensorrt path
        PATHS "${PROJECT_SOURCE_DIR}/third_party/TensorRT/" # or custom tensorrt path
        )
list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)

# Include dir
foreach (search ${_TensorRT_SEARCHES})
    find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h NvInferPlugin.h ${${search}} PATH_SUFFIXES include)
endforeach ()

if (NOT TensorRT_LIBRARY)
    foreach (search ${_TensorRT_SEARCHES})
        find_library(TensorRT_LIBRARY NAMES nvinfer ${${search}} PATH_SUFFIXES lib)
    endforeach ()
endif ()

if (NOT TensorRT_PLUGIN_LIBRARY)
    foreach (search ${_TensorRT_SEARCHES})
        find_library(TensorRT_PLUGIN_LIBRARY NAMES nvinfer_plugin ${${search}} PATH_SUFFIXES lib)
    endforeach ()
endif ()



if (TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif ()


include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT REQUIRED_VARS TensorRT_LIBRARY TensorRT_PLUGIN_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION_STRING)
message(STATUS "TensorRT_LIBRARY: ${TensorRT_LIBRARY}")
message(STATUS "TensorRT_PLUGIN_LIBRARY: ${TensorRT_PLUGIN_LIBRARY}")
message(STATUS "TensorRT_INCLUDE_DIR: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT: ${TensorRT_VERSION_STRING}")
if (TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

    if (NOT TensorRT_LIBRARIES)
        set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
    endif ()

    if (NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
        set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
        set_property(TARGET TensorRT::TensorRT APPEND PROPERTY VERSION "${TensorRT_VERSION_STRING}")
    endif ()

    if (NOT TARGET TensorRT::Plugin)
        add_library(TensorRT::Plugin UNKNOWN IMPORTED)
        set_target_properties(TensorRT::Plugin PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
        set_property(TARGET TensorRT::Plugin APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_PLUGIN_LIBRARY}")
        set_property(TARGET TensorRT::Plugin APPEND PROPERTY VERSION "${TensorRT_VERSION_STRING}")
    endif ()
endif ()


