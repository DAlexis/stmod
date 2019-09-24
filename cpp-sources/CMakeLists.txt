cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project("stmod" CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -pg -g -lprofiler -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -DNDEBUG -funroll-loops -ffast-math -lprofiler")

if (NOT COMMAND add_functional_test)
	add_custom_target(run-functional-tests)
	macro(add_functional_test test_script script_argument)
	    set(command_to_run ./${test_script})
	    set(target_name ${test_script}_${script_argument}_run_func_test)
	    string(REPLACE "/" "_" target_name ${target_name})
	  	add_custom_target(${target_name}
	    	COMMAND ${command_to_run} ${script_argument}
	      	WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
	  	add_dependencies(run-functional-tests ${target_name})
	endmacro()
endif()

## This macro creates variable ${PROJECT_NAME}_INCLUDE_DIRS with list of 
# include directories exported by project
macro(export_include_dirs)
    set(${PROJECT_NAME}_INCLUDE_DIRS
        ${ARGV}
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
endmacro()

# This macro creates variable ${PROJECT_NAME}_LIBRARY with list of libraries exported 
# by project
macro(export_libs)
    set(${PROJECT_NAME}_LIBRARY
        ${ARGV}
        CACHE INTERNAL "${PROJECT_NAME}: used and exported libs" FORCE)
endmacro()

add_subdirectory(stmod)
add_subdirectory(model-runner)

# To enable ctest usage
enable_testing()
include(detect-gtest.cmake)
if(GTEST_FOUND)
    add_subdirectory(unit-tests)
    #add_subdirectory(functional-tests)
endif()

