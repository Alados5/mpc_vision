# What version of CMake is needed?
cmake_minimum_required(VERSION 2.8.3)

# The name of this package.
project(mpc_vision)

# Find the catkin build system, and any other packages on which we depend.
find_package(catkin REQUIRED COMPONENTS
    roscpp
    geometry_msgs
    std_msgs
    message_generation
    actionlib
    actionlib_msgs
    trajectory_msgs
    control_msgs
    sensor_msgs
    tf2
    tf2_ros
)
find_package(Eigen3 REQUIRED)

## Generate MESSAGES in the 'msg' folder
add_message_files(
    FILES
    ClothMesh.msg
    SOMstate.msg
)

## Generate services in the 'srv' folder
#add_service_files(
#   FILES
   #CheckInitialPositions.srv
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   actionlib_msgs
   trajectory_msgs
   control_msgs
   sensor_msgs
   geometry_msgs
 )

# Declare our catkin package.
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demopkg
   CATKIN_DEPENDS roscpp std_msgs message_runtime
#  DEPENDS system_lib
)

# Specify locations of header files.
include_directories(../include)
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(EIGEN3_INCLUDE_DIR)

# Create executable.
file(GLOB CALIB_SOURCES "src/general_functions.cpp" "src/calibration.cpp")
add_executable(calibration ${CALIB_SOURCES})
file(GLOB UPDATE_SOURCES "src/general_functions.cpp" "src/update_som.cpp")
add_executable(update_som ${UPDATE_SOURCES})

# Link necessary libraries.
target_link_libraries(calibration ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
target_link_libraries(update_som ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_dependencies(calibration ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(update_som ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

# Set flag: C++11 needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")








