find_path(ACADOS_INCLUDE_DIRS
    NAMES ocp_nlp_interface.h
    HINTS /usr/include /usr/local/include/acados_c)

find_library(ACADOS_LIBRARIES
    NAMES acados
    HINTS /usr/local/lib /usr/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACADOS DEFAULT_MSG ACADOS_LIBRARIES)
