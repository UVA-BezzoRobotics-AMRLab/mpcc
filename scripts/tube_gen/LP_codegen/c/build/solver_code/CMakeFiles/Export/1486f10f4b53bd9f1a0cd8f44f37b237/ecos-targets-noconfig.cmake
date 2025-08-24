#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ecos::ecos" for configuration ""
set_property(TARGET ecos::ecos APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ecos::ecos PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libecos.so"
  IMPORTED_SONAME_NOCONFIG "libecos.so"
  )

list(APPEND _cmake_import_check_targets ecos::ecos )
list(APPEND _cmake_import_check_files_for_ecos::ecos "${_IMPORT_PREFIX}/lib/libecos.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
