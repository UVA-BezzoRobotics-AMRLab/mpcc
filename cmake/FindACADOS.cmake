find_path(ACADOS_INCLUDE_DIRS
    NAMES ocp_nlp_interface.h
    HINTS $ENV{ACADOS_SOURCE_DIR}/include/acados_c /usr/local/include/acados_c /usr/include/acados_c)

find_library(ACADOS_LIBRARIES
    NAMES acados
    HINTS $ENV{ACADOS_SOURCE_DIR}/lib /usr/local/lib /usr/lib)

# if acados_include_dirs has acados_c after /include, remove acados_c from path
string(REGEX REPLACE "/acados_c$" "" ACADOS_INCLUDE_DIRS "${ACADOS_INCLUDE_DIRS}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACADOS DEFAULT_MSG ACADOS_LIBRARIES)
