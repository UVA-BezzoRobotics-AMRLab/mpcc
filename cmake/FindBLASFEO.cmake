find_path(BLASFEO_INCLUDE_DIRS
    NAMES blasfeo_common.h
    HINTS $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include /usr/local/include/blasfeo /usr/include/blasfeo)

find_library(BLASFEO_LIBRARIES
    NAMES blasfeo
    HINTS $ENV{ACADOS_SOURCE_DIR}/lib /usr/local/lib /usr/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASFEO DEFAULT_MSG BLASFEO_LIBRARIES)


