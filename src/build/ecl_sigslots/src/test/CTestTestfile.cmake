# CMake generated Testfile for 
# Source directory: /home/ri-one/ros2_ws/src/turtlebot2_ros2/ecl_core/ecl_sigslots/src/test
# Build directory: /home/ri-one/ros2_ws/src/build/ecl_sigslots/src/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ecl_test_sigslots "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/ri-one/ros2_ws/src/build/ecl_sigslots/test_results/ecl_sigslots/ecl_test_sigslots.gtest.xml" "--package-name" "ecl_sigslots" "--output-file" "/home/ri-one/ros2_ws/src/build/ecl_sigslots/ament_cmake_gtest/ecl_test_sigslots.txt" "--command" "/home/ri-one/ros2_ws/src/build/ecl_sigslots/src/test/ecl_test_sigslots" "--gtest_output=xml:/home/ri-one/ros2_ws/src/build/ecl_sigslots/test_results/ecl_sigslots/ecl_test_sigslots.gtest.xml")
set_tests_properties(ecl_test_sigslots PROPERTIES  LABELS "gtest" REQUIRED_FILES "/home/ri-one/ros2_ws/src/build/ecl_sigslots/src/test/ecl_test_sigslots" TIMEOUT "60" WORKING_DIRECTORY "/home/ri-one/ros2_ws/src/build/ecl_sigslots/src/test" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_gtest/cmake/ament_add_gtest_test.cmake;86;ament_add_test;/opt/ros/humble/share/ament_cmake_gtest/cmake/ament_add_gtest.cmake;93;ament_add_gtest_test;/home/ri-one/ros2_ws/src/turtlebot2_ros2/ecl_core/ecl_sigslots/src/test/CMakeLists.txt;6;ament_add_gtest;/home/ri-one/ros2_ws/src/turtlebot2_ros2/ecl_core/ecl_sigslots/src/test/CMakeLists.txt;20;ecl_sigslots_add_gtest;/home/ri-one/ros2_ws/src/turtlebot2_ros2/ecl_core/ecl_sigslots/src/test/CMakeLists.txt;0;")
subdirs("../../gtest")
