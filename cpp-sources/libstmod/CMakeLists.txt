cmake_minimum_required(VERSION 2.8)

project(stmod)

find_package(VTK REQUIRED)
find_package(deal.II REQUIRED)
find_package(TBB REQUIRED)
#include(${VTK_USE_FILE})

set(LIB_SOURCE
    ${PROJECT_SOURCE_DIR}/src/fe-common.cpp
    ${PROJECT_SOURCE_DIR}/src/fe-sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/field-output.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions/fraction-base.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions/fraction.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/e.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/heat-power.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/electric-potential.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/reaction-constants.cpp
    ${PROJECT_SOURCE_DIR}/src/full-models/model-one.cpp
    ${PROJECT_SOURCE_DIR}/src/grid/grid.cpp
    ${PROJECT_SOURCE_DIR}/src/grid/mesh-refiner.cpp
    ${PROJECT_SOURCE_DIR}/src/matgen.cpp
    ${PROJECT_SOURCE_DIR}/src/output/output.cpp
    ${PROJECT_SOURCE_DIR}/src/phys-consts.cpp
    ${PROJECT_SOURCE_DIR}/src/tensors.cpp
    ${PROJECT_SOURCE_DIR}/src/time/time-iteration.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)

set(LIB_HPP
    ${PROJECT_SOURCE_DIR}/stmod/fe-common.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fe-sampler.hpp
    ${PROJECT_SOURCE_DIR}/stmod/field-output.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions/fraction.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/e.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/heat-power.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/electric-potential.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/reaction-constants.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions/secondary-value.hpp
    ${PROJECT_SOURCE_DIR}/stmod/full-models/model-one.hpp
    ${PROJECT_SOURCE_DIR}/stmod/grid/grid.hpp
    ${PROJECT_SOURCE_DIR}/stmod/grid/mesh-based.hpp
    ${PROJECT_SOURCE_DIR}/stmod/grid/mesh-refiner.hpp
    ${PROJECT_SOURCE_DIR}/stmod/matgen.hpp
    ${PROJECT_SOURCE_DIR}/stmod/output/output.hpp
    ${PROJECT_SOURCE_DIR}/stmod/output/output-provider.hpp
    ${PROJECT_SOURCE_DIR}/stmod/phys-consts.hpp
    ${PROJECT_SOURCE_DIR}/stmod/tensors.hpp
    ${PROJECT_SOURCE_DIR}/stmod/time/time-iterable.hpp
    ${PROJECT_SOURCE_DIR}/stmod/time/time-iteration.hpp
    ${PROJECT_SOURCE_DIR}/stmod/utils.hpp
    ${PROJECT_SOURCE_DIR}/stmod/variable.hpp
)

add_library(${PROJECT_NAME} STATIC ${LIB_SOURCE} ${LIB_HPP})
target_link_libraries(${PROJECT_NAME} PUBLIC profiler tbb ${DEAL_II_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${DEAL_II_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_rvalue_references)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_lambdas)
