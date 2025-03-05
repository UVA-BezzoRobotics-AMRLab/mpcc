cmake_minimum_required(VERSION 3.0.2)
project(mpcc C CXX)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++17 -O3)

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
  grid_map_ros
  costmap_2d
  amrl_logging
)

set(CVXPYGEN_OUTPUT ${CMAKE_SOURCE_DIR}/scripts/tube_gen/LP_codegen/c)
add_subdirectory(${CVXPYGEN_OUTPUT})

## System dependencies are found with CMake's conventions
find_package(Python3 REQUIRED)
find_package(ACADOS REQUIRED)
find_package(BLASFEO REQUIRED)
find_package(HPIPM REQUIRED)
find_package(Eigen3 REQUIRED COMPONENT system)

# if acados_include_dirs has acados_c after /include, remove acados_c from path
string(REGEX REPLACE "/acados_c$" "" ACADOS_INCLUDE_DIRS "${ACADOS_INCLUDE_DIRS}")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mpcc
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
  ${HPIPM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${ACADOS_INCLUDE_DIRS}
  ${BLASFEO_INCLUDE_DIRS}
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
  ${ACADOS_OUTPUT_DIR}/libacados_sim_solver_unicycle_model_mpcc.so
)

set(ACADOS_OUTPUT_FILES ${ACADOS_OUTPUT_SRC} ${ACADOS_OUTPUT_HEADERS} ${ACADOS_OUTPUT_LIB})

set(ACADOS_SCRIPTS 
  ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_model.py
  ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_ocp_horizon.py
)

add_custom_command(
  OUTPUT ${ACADOS_OUTPUT_FILES}
  COMMAND echo "y" | ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc/mpcc_ocp_horizon.py --yaml ${CMAKE_SOURCE_DIR}/params/mpcc.yaml
  DEPENDS ${ACADOS_SCRIPTS} ${CMAKE_SOURCE_DIR}/params/mpcc.yaml
  COMMENT "Building MPC code with Acados"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/unicycle_mpcc
  VERBATIM
)

add_custom_target(
  GenerateAcadosCode
  DEPENDS ${ACADOS_OUTPUT_FILES} ${ACADOS_SCRIPTS} ${CMAKE_SOURCE_DIR}/params/mpcc.yaml
)

# add_executable(test_reparam src/test.cpp)
# target_link_libraries(test_reparam
#   ${catkin_LIBRARIES}
#   ${EIGEN3_LIBRARIES}
# )

add_library(cvxpy_lib STATIC ${cpg_src})
target_compile_options(cvxpy_lib PRIVATE -O3 -w)
target_include_directories(cvxpy_lib PRIVATE ${cpg_include})

add_executable(mpcc_ros 
  src/mpcc/logger.cpp
  src/mpcc/mpcc_node.cpp
  src/mpcc/mpcc_ros.cpp
  src/mpcc/mpcc_core.cpp
  src/mpcc/mpcc_acados.cpp
)
target_link_libraries(mpcc_ros
  ${HPIPM_LIBRARIES}
  ${catkin_LIBRARIES}
  ${ACADOS_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${ACADOS_OUTPUT_LIB}
  ${BLASFEO_LIBRARIES}
  cvxpy_lib
)
target_compile_options(mpcc_ros PRIVATE -std=c++17 -O3)
add_dependencies(mpcc_ros GenerateAcadosCode)
add_dependencies(mpcc_ros cvxpy_lib)
target_include_directories(mpcc_ros PRIVATE ${ACADOS_OUTPUT_DIR} ${cpg_include})

add_executable(publish_pf_pose src/utils/publish_pf_pose.cpp)
target_link_libraries(publish_pf_pose
  ${catkin_LIBRARIES}
)

add_executable(local_costmap src/utils/local_costmap.cpp)
target_link_libraries(local_costmap
    ${catkin_LIBRARIES}
)
