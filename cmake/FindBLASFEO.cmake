find_path(BLASFEO_INCLUDE_DIRS
    NAMES blasfeo_common.h
    HINTS /usr/include /usr/local/include/blasfeo/include)

find_library(BLASFEO_LIBRARIES
    NAMES libblasfeo.so
    HINTS /usr/local/lib /usr/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASFEO DEFAULT_MSG BLASFEO_LIBRARIES)


