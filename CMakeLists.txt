cmake_minimum_required(VERSION 3.0.2)
project(mpcc C CXX)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++17 -O3)

# add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(AcadosUtils)

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
# string(REGEX REPLACE "/acados_c$" "" ACADOS_INCLUDE_DIRS "${ACADOS_INCLUDE_DIRS}")

message("ACADOS_INCLUDE_DIRS: ${ACADOS_INCLUDE_DIRS}")

add_service_files(
  FILES
  QuerySAC.srv
  QuerySACDI.srv
)

generate_messages(
      DEPENDENCIES 
      nav_msgs
      geometry_msgs 
      trajectory_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
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
  ${cpg_include}
)


set(ACADOS_MODELS
  unicycle_model_mpcc
  double_integrator_mpcc
)

set(ACADOS_OUTPUT_DIRS "")
set(ACADOS_OUTPUT_SRC "")
set(ACADOS_OUTPUT_HEADERS "")
set(ACADOS_OUTPUT_LIBS "")
set(ACADOS_TARGETS "")
foreach(model ${ACADOS_MODELS})
  set(${model}_SCRIPTS 
    ${CMAKE_SOURCE_DIR}/scripts/${model}/mpcc_model.py
    ${CMAKE_SOURCE_DIR}/scripts/${model}/mpcc_ocp.py
  )
  list(APPEND ACADOS_OUTPUT_DIRS ${CMAKE_SOURCE_DIR}/scripts/${model}/c_generated_code)

  collect_acados_generated_code(${model})
  list(APPEND ACADOS_OUTPUT_SRC ${${model}_SRC})
  list(APPEND ACADOS_OUTPUT_HEADERS ${${model}_HEADERS})
  list(APPEND ACADOS_OUTPUT_LIBS ${${model}_LIBS})

  # add model specific output to a variable called ${model}_output_files

  add_custom_command(
    OUTPUT ${${model}_OUTPUT_FILES}
    COMMAND echo "y" | ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/${model}/mpcc_ocp.py --yaml ${CMAKE_SOURCE_DIR}/params/${model}.yaml
    DEPENDS ${ACADOS_SCRIPTS} ${CMAKE_SOURCE_DIR}/params/${model}.yaml
    COMMENT "Building MPC code with Acados"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/${model}
    VERBATIM
  )

  message("scripts: ${${model}_SCRIPTS}")
  message("outfiles: ${${model}_LIBS}")

  add_custom_target(
    GenerateAcadosCode_${model}
    DEPENDS ${${model}_OUTPUT_FILES} ${${model}_SCRIPTS} ${CMAKE_SOURCE_DIR}/params/${model}.yaml
  )
  list(APPEND ACADOS_TARGETS GenerateAcadosCode_${model})
endforeach()

add_library(cvxpy_lib STATIC ${cpg_src})
target_compile_options(cvxpy_lib PRIVATE -O3 -w)
target_include_directories(cvxpy_lib PRIVATE ${cpg_include})

add_executable(mpcc_ros 
  src/mpcc/logger.cpp
  src/mpcc/mpcc_node.cpp
  src/mpcc/mpcc_ros.cpp
  src/mpcc/mpcc_core.cpp
  src/mpcc/mpcc_acados.cpp
  src/mpcc/mpcc_di_acados.cpp
)
target_link_libraries(mpcc_ros
  ${HPIPM_LIBRARIES}
  ${catkin_LIBRARIES}
  ${ACADOS_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${ACADOS_OUTPUT_LIBS}
  ${BLASFEO_LIBRARIES}
  cvxpy_lib
)
target_compile_options(mpcc_ros PRIVATE -std=c++17 -O3)
add_dependencies(mpcc_ros ${ACADOS_TARGETS})
add_dependencies(mpcc_ros cvxpy_lib)
target_include_directories(mpcc_ros PRIVATE ${ACADOS_OUTPUT_DIRS} ${cpg_include})

add_executable(publish_pf_pose src/utils/publish_pf_pose.cpp)
target_link_libraries(publish_pf_pose
  ${catkin_LIBRARIES}
)

add_executable(local_costmap src/utils/local_costmap.cpp)
target_link_libraries(local_costmap
    ${catkin_LIBRARIES}
)
