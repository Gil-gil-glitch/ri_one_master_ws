# Install script for directory: /home/ri-one/ros2_ws/src/turtlebot2_ros2/ecl_core/ecl_sigslots/src/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ri-one/ros2_ws/src/install/ecl_sigslots")
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots" TYPE EXECUTABLE FILES "/home/ri-one/ros2_ws/src/build/ecl_sigslots/src/examples/demo_sigslots")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots"
         OLD_RPATH "/home/ri-one/ros2_ws/src/install/ecl_threads/lib:/home/ri-one/ros2_ws/src/install/ecl_time/lib:/home/ri-one/ros2_ws/src/install/ecl_exceptions/lib:/home/ri-one/ros2_ws/src/install/ecl_time_lite/lib:/home/ri-one/ros2_ws/src/install/ecl_errors/lib:/home/ri-one/ros2_ws/src/install/ecl_type_traits/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots" TYPE EXECUTABLE FILES "/home/ri-one/ros2_ws/src/build/ecl_sigslots/src/examples/demo_sigslots_manager")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager"
         OLD_RPATH "/home/ri-one/ros2_ws/src/install/ecl_threads/lib:/home/ri-one/ros2_ws/src/install/ecl_time/lib:/home/ri-one/ros2_ws/src/install/ecl_exceptions/lib:/home/ri-one/ros2_ws/src/install/ecl_time_lite/lib:/home/ri-one/ros2_ws/src/install/ecl_errors/lib:/home/ri-one/ros2_ws/src/install/ecl_type_traits/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ecl_sigslots/demo_sigslots_manager")
    endif()
  endif()
endif()

