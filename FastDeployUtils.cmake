CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

function(get_openvino_libs OPENVINO_RUNTIME_DIR)
  set(LIB_LIST "") 
  find_library(OPENVINO_LIB openvino ${OPENVINO_RUNTIME_DIR}/lib/ NO_DEFAULT_PATH)
  list(APPEND LIB_LIST ${OPENVINO_LIB})

  find_package(TBB PATHS "${OPENVINO_RUNTIME_DIR}/3rdparty/tbb")
  if (TBB_FOUND)
    list(APPEND LIB_LIST ${TBB_IMPORTED_TARGETS})
  else()
    # TODO(zhoushunjie): Use openvino with tbb on linux in future.
    set(OMP_LIB "${OPENVINO_RUNTIME_DIR}/3rdparty/omp/lib/libiomp5.so")
    list(APPEND LIB_LIST ${OMP_LIB})
  endif()
  set(OPENVINO_LIBS ${LIB_LIST} PARENT_SCOPE)
endfunction()