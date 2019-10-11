cmake_minimum_required(VERSION 2.8)

project(stmod)

find_package(VTK REQUIRED)
find_package(deal.II REQUIRED)
#include(${VTK_USE_FILE})
message(STATUS ${DEAL_II_INCLUDE_DIRS})

set(LIB_SOURCE
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-solver.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-grid.cpp
    ${PROJECT_SOURCE_DIR}/src/fe-sampler.cpp
)

set(LIB_HPP
    ${PROJECT_SOURCE_DIR}/include/utils.hpp
    ${PROJECT_SOURCE_DIR}/include/poisson-solver.hpp
    ${PROJECT_SOURCE_DIR}/include/poisson-grid.hpp
    ${PROJECT_SOURCE_DIR}/include/fe-sampler.hpp
)

# Include dirs that are used and will be exported 
set(${PROJECT_NAME}_USED_INCDIRS
    ${PROJECT_SOURCE_DIR}
    ${DEAL_II_INCLUDE_DIRS}
)

include_directories(${${PROJECT_NAME}_USED_INCDIRS})

add_library(${PROJECT_NAME} STATIC ${LIB_SOURCE} ${LIB_HPP})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_rvalue_references)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_lambdas)

# Include directories that should be added with this project
export_include_dirs(
    ${${PROJECT_NAME}_USED_INCDIRS}
)

# Libraries this projects needs to be linked with
export_libs(
    ${PROJECT_NAME}
    ${DEAL_II_LIBRARIES}
)
