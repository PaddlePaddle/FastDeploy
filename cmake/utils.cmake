# This function comes from https://blog.csdn.net/yindongjie1221/article/details/90614261
function(redefine_file_macro targetname)
    get_target_property(source_files "${targetname}" SOURCES)
    foreach(sourcefile ${source_files})
        get_property(defs SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS)
        get_filename_component(filepath "${sourcefile}" ABSOLUTE)
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" relpath ${filepath})
        list(APPEND defs "__REL_FILE__=\"${relpath}\"")
        set_property(
            SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS ${defs}
            )
    endforeach()
endfunction()

function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS ${filename})
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp")
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
  endif()
  message("Decompress file ${filename} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
endfunction()

function(get_openvino_libs OPENVINO_RUNTIME_DIR)
  set(LIB_LIST "")
  find_library(OPENVINO_LIB openvino PATHS ${OPENVINO_RUNTIME_DIR}/lib/ ${OPENVINO_RUNTIME_DIR}/lib/intel64 NO_DEFAULT_PATH)
  list(APPEND LIB_LIST ${OPENVINO_LIB})

  find_package(TBB PATHS ${OPENVINO_RUNTIME_DIR}/3rdparty/tbb)
  if (TBB_FOUND)
    list(APPEND LIB_LIST ${TBB_IMPORTED_TARGETS})
  else()
    # TODO(zhoushunjie): Use openvino with tbb on linux in future.
    set(OMP_LIB "${OPENVINO_RUNTIME_DIR}/3rdparty/omp/lib/libiomp5.so")
    list(APPEND LIB_LIST ${OMP_LIB})
  endif()
  set(OPENVINO_LIBS ${LIB_LIST} PARENT_SCOPE)
endfunction()

function(remove_duplicate_libraries libraries)
  list(LENGTH ${libraries} lib_length)
  set(libraries_temp "")
  set(full_libraries "")
  foreach(lib_path ${${libraries}})
    get_filename_component(lib_name ${lib_path} NAME)
    list(FIND libraries_temp ${lib_name} lib_idx)
    if (${lib_idx} EQUAL -1)
      list(APPEND libraries_temp ${lib_name})
      list(APPEND full_libraries ${lib_path})
    endif()
  endforeach()
  set(${libraries} ${full_libraries} PARENT_SCOPE)
endfunction()
