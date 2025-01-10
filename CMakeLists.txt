cmake_minimum_required(VERSION 3.0.2)
project(uav_mpc)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17 -O3)

# add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  trajectory_msgs
  distance_map_core
  grid_map_ros
)

## System dependencies are found with CMake's conventions
find_package(Python3 REQUIRED)
find_package(ACADOS REQUIRED)
find_package(BLASFEO REQUIRED)
find_package(HPIPM REQUIRED)
find_package(nlopt REQUIRED)
find_package(Eigen3 REQUIRED COMPONENT system)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES uav_mpc
#  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs trajectory_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  third_party/autodiff
  ${catkin_INCLUDE_DIRS}
  ${ACADOS_INCLUDE_DIRS}
  ${HPIPM_INCLUDE_DIRS}
  ${BLASFEO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(ACADOS_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/c_generated_code)

set(ACADOS_OUTPUT_SRC
  ${ACADOS_OUTPUT_DIR}/acados_solver_unicycle_model_mpcc.c
  ${ACADOS_OUTPUT_DIR}/acados_sim_solver_unicycle_model_mpcc.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_model/unicycle_model_mpcc_expl_ode_fun.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_model/unicycle_model_mpcc_expl_vde_adj.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_model/unicycle_model_mpcc_expl_vde_forw.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_model/unicycle_model_mpcc_expl_ode_hess.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_fun.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_0_fun.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_e_fun.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_fun_jac.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_0_fun_jac.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_e_fun_jac.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_fun_jac_hess.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_0_fun_jac_hess.c
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost_ext_cost_e_fun_jac_hess.c
)

set(ACADOS_OUTPUT_HEADERS
  ${ACADOS_OUTPUT_DIR}/acados_solver_unicycle_model_mpcc.h
  ${ACADOS_OUTPUT_DIR}/acados_sim_solver_unicycle_model_mpcc.h
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_cost/unicycle_model_mpcc_cost.h
  ${ACADOS_OUTPUT_DIR}/unicycle_model_mpcc_model/unicycle_model_mpcc_model.h
)

set(ACADOS_OUTPUT_LIB
  ${ACADOS_OUTPUT_DIR}/libacados_ocp_solver_unicycle_model_mpcc.so
)

set(ACADOS_OUTPUT_FILES ${ACADOS_OUTPUT_SRC} ${ACADOS_OUTPUT_HEADERS} ${ACADOS_OUTPUT_LIB})

set(ACADOS_SCRIPTS 
  ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_model.py
  ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_ocp_horizon.py
)

add_custom_command(
  OUTPUT ${ACADOS_OUTPUT_FILES}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_ocp_horizon.py
  DEPENDS ${ACADOS_SCRIPTS}
  COMMENT "Building MPC code with Acados"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc
  VERBATIM
)

add_custom_target(
  GenerateAcadosCode
  DEPENDS ${ACADOS_OUTPUT_FILES} ${ACADOS_SCRIPTS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/uav_mpc.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# set(SOURCES
#     src/unicycle_mpcc.cpp
#     include/uav_mpc/unicycle_mpcc/acados_solver_unicycle_model.c
#     include/uav_mpc/unicycle_mpcc/unicycle_model_model/unicycle_model_impl_dae_fun.c
#     include/uav_mpc/unicycle_mpcc/unicycle_model_model/unicycle_model_impl_dae_fun_jac_x_xdot_u.c
#     include/uav_mpc/unicycle_mpcc/unicycle_model_model/unicycle_model_impl_dae_fun_jac_x_xdot_u_z.c
#     include/uav_mpc/unicycle_mpcc/unicycle_model_model/unicycle_model_impl_dae_fun_jac_x_xdot_z.c
#     include/uav_mpc/unicycle_mpcc/unicycle_model_model/unicycle_model_impl_dae_jac_x_xdot_u_z.c
# )

# add_executable(unicycle_mpcc ${SOURCES})
# target_link_libraries(unicycle_mpcc
#   ${ACADOS_LIBRARIES}
#   ${BLASFEO_LIBRARIES}
#   ${HPIPM_LIBRARIES}
#   ${catkin_LIBRARIES}
#   ${EIGEN3_LIBRARIES}
#   m
# )

# add_executable(casadi_mpc src/mpcc_casadi_impl.cpp)
# target_link_libraries(casadi_mpc
#   ${catkin_LIBRARIES}
#   ${EIGEN3_LIBRARIES}
#   ${NLOPT_LIBRARIES}
# )

add_executable(test_reparam src/test.cpp)
target_link_libraries(test_reparam
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

add_executable(mpcc_ros 
  src/mpcc_node.cpp
  src/mpcc_ros.cpp
  src/mpcc_core.cpp
  src/mpcc_impl.cpp
  # src/mpcc_nlopt_impl.cpp
)
target_link_libraries(mpcc_ros
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  # ${NLOPT_LIBRARIES}
  ipopt
)

add_executable(mpcc_acados src/mpcc_acados.cpp)
target_link_libraries(mpcc_acados
  ${ACADOS_LIBRARIES}
  ${BLASFEO_LIBRARIES}
  ${HPIPM_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${ACADOS_OUTPUT_LIB}
)
add_dependencies(mpcc_acados GenerateAcadosCode)
target_include_directories(mpcc_acados PRIVATE ${ACADOS_OUTPUT_DIR})
