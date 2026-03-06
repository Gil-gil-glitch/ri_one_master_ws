# Install script for directory: /home/ri-one/ros2_ws/src/turtlebot2_ros2/kobuki_core/src/tools

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ri-one/ros2_ws/src/install/kobuki_core")
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/ri-one/ros2_ws/src/build/kobuki_core/src/tools/kobuki-version-info")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info"
         OLD_RPATH "/home/ri-one/ros2_ws/src/build/kobuki_core/src/driver:/home/ri-one/ros2_ws/src/install/ecl_devices/lib:/home/ri-one/ros2_ws/src/install/ecl_mobile_robot/lib:/home/ri-one/ros2_ws/src/install/ecl_geometry/lib:/home/ri-one/ros2_ws/src/install/ecl_linear_algebra/lib:/home/ri-one/ros2_ws/src/install/ecl_formatters/lib:/home/ri-one/ros2_ws/src/install/ecl_threads/lib:/home/ri-one/ros2_ws/src/install/ecl_type_traits/lib:/home/ri-one/ros2_ws/src/install/ecl_time/lib:/home/ri-one/ros2_ws/src/install/ecl_exceptions/lib:/home/ri-one/ros2_ws/src/install/ecl_time_lite/lib:/home/ri-one/ros2_ws/src/install/ecl_errors/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-version-info")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/ri-one/ros2_ws/src/build/kobuki_core/src/tools/kobuki-simple-keyop")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop"
         OLD_RPATH "/home/ri-one/ros2_ws/src/build/kobuki_core/src/driver:/home/ri-one/ros2_ws/src/install/ecl_devices/lib:/home/ri-one/ros2_ws/src/install/ecl_mobile_robot/lib:/home/ri-one/ros2_ws/src/install/ecl_geometry/lib:/home/ri-one/ros2_ws/src/install/ecl_linear_algebra/lib:/home/ri-one/ros2_ws/src/install/ecl_formatters/lib:/home/ri-one/ros2_ws/src/install/ecl_threads/lib:/home/ri-one/ros2_ws/src/install/ecl_type_traits/lib:/home/ri-one/ros2_ws/src/install/ecl_time/lib:/home/ri-one/ros2_ws/src/install/ecl_exceptions/lib:/home/ri-one/ros2_ws/src/install/ecl_time_lite/lib:/home/ri-one/ros2_ws/src/install/ecl_errors/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/kobuki-simple-keyop")
    endif()
  endif()
endif()

