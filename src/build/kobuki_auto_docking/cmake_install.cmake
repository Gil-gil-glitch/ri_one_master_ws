# Install script for directory: /home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ri-one/ros2_ws/src/install/kobuki_auto_docking")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking" TYPE EXECUTABLE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/kobuki_auto_docking_node")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking/kobuki_auto_docking_node")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/libkobuki_auto_docking.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so"
         OLD_RPATH "/home/ri-one/ros2_ws/src/install/kobuki_core/lib:/home/ri-one/ros2_ws/src/install/kobuki_ros_interfaces/lib:/opt/ros/humble/lib:/home/ri-one/ros2_ws/src/install/ecl_devices/lib:/home/ri-one/ros2_ws/src/install/ecl_mobile_robot/lib:/home/ri-one/ros2_ws/src/install/ecl_geometry/lib:/home/ri-one/ros2_ws/src/install/ecl_linear_algebra/lib:/home/ri-one/ros2_ws/src/install/ecl_formatters/lib:/home/ri-one/ros2_ws/src/install/ecl_threads/lib:/home/ri-one/ros2_ws/src/install/ecl_type_traits/lib:/home/ri-one/ros2_ws/src/install/ecl_time/lib:/home/ri-one/ros2_ws/src/install/ecl_exceptions/lib:/home/ri-one/ros2_ws/src/install/ecl_time_lite/lib:/home/ri-one/ros2_ws/src/install/ecl_errors/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkobuki_auto_docking.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kobuki_auto_docking" TYPE DIRECTORY FILES "/home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking/include/kobuki_auto_docking/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE DIRECTORY FILES
    "/home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking/launch"
    "/home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking/config"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/kobuki_auto_docking" TYPE PROGRAM FILES "/home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking/scripts/activate.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/kobuki_auto_docking")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/kobuki_auto_docking")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking/environment" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking/environment" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_index/share/ament_index/resource_index/packages/kobuki_auto_docking")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rclcpp_components" TYPE FILE FILES "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_index/share/ament_index/resource_index/rclcpp_components/kobuki_auto_docking")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking/cmake" TYPE FILE FILES
    "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_core/kobuki_auto_dockingConfig.cmake"
    "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/ament_cmake_core/kobuki_auto_dockingConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/kobuki_auto_docking" TYPE FILE FILES "/home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_ros/kobuki_auto_docking/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ri-one/ros2_ws/src/build/kobuki_auto_docking/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
