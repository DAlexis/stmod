cmake_minimum_required(VERSION 2.8)

project(stmod)

find_package(VTK REQUIRED)
find_package(deal.II REQUIRED)
#include(${VTK_USE_FILE})

set(LIB_SOURCE
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-solver.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-grid.cpp
    ${PROJECT_SOURCE_DIR}/src/fe-sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/field-output.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/e.cpp
)

set(LIB_HPP
    ${PROJECT_SOURCE_DIR}/stmod/utils.hpp
    ${PROJECT_SOURCE_DIR}/stmod/poisson-solver.hpp
    ${PROJECT_SOURCE_DIR}/stmod/poisson-grid.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fe-sampler.hpp
    ${PROJECT_SOURCE_DIR}/stmod/field-output.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/e.hpp
)

add_library(${PROJECT_NAME} STATIC ${LIB_SOURCE} ${LIB_HPP})
target_link_libraries(${PROJECT_NAME} PUBLIC profiler dsiterpp ${DEAL_II_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${DEAL_II_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_rvalue_references)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_lambdas)
