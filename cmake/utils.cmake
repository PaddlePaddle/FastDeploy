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
