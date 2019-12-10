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

include(detect-gtest.cmake)
add_subdirectory(dsiterpp/cpp-sources)
add_subdirectory(libstmod)
add_subdirectory(model-runner)

# To enable ctest usage
enable_testing()

if(GTEST_FOUND)
    add_subdirectory(unit-tests)
    #add_subdirectory(functional-tests)
endif()

