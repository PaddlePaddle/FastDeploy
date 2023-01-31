find_package(Doxygen QUIET)
if(NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen is needed to build the documentation.")
endif()

# TODO: add config file
set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(doxyfile    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(doxygen_output_dir ${CMAKE_CURRENT_BINARY_DIR}/doxygen)

configure_file(${doxyfile_in} ${doxyfile} @ONLY)

file(MAKE_DIRECTORY ${doxygen_output_dir})

add_custom_target(Doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable")

set(sphinx_source ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(sphinx_build ${CMAKE_CURRENT_BINARY_DIR}/build/latest)
set(doxygen_xml_builddir ${doxygen_output_dir}/xml)

add_custom_target(Sphinx ALL
    COMMAND ${SPHINX_EXECUTABLE} -b html
    -Dbreathe_projects.dlpack=${doxygen_xml_builddir}
    ${sphinx_source} ${sphinx_build} -WT --keep-going
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx"
    VERBATIM)

install(DIRECTORY ${sphinx_build} DESTINATION share/${PROJECT_NAME}/docs)
