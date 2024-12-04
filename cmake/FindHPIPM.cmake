find_path(HPIPM_INCLUDE_DIRS
    NAMES hpipm_common.h
    HINTS /usr/include /usr/local/include/hpipm/include)

find_library(HPIPM_LIBRARIES
    NAMES libhpipm.so
    HINTS /usr/local/lib /usr/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HPIPM DEFAULT_MSG HPIPM_LIBRARIES)
