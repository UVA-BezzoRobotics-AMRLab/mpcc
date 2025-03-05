find_path(HPIPM_INCLUDE_DIRS
    NAMES hpipm_common.h
    HINTS $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include /usr/local/include/hpipm /usr/include/hpipm)

find_library(HPIPM_LIBRARIES
    NAMES hpipm
    HINTS $ENV{ACADOS_SOURCE_DIR}/lib /usr/local/lib /usr/lib)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HPIPM DEFAULT_MSG HPIPM_LIBRARIES)
