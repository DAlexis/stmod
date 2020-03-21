    cmake_minimum_required(VERSION 2.8)

project(stmod)

find_package(VTK REQUIRED)
find_package(deal.II REQUIRED)
#include(${VTK_USE_FILE})

set(LIB_SOURCE
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-solver.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-solver-adaptor.cpp
    ${PROJECT_SOURCE_DIR}/src/poisson-grid.cpp
    ${PROJECT_SOURCE_DIR}/src/fe-sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/field-output.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-base.cpp
    ${PROJECT_SOURCE_DIR}/src/matgen.cpp
    ${PROJECT_SOURCE_DIR}/src/tensors.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh-output.cpp
    ${PROJECT_SOURCE_DIR}/src/fe-common.cpp
    ${PROJECT_SOURCE_DIR}/src/lin-eq-solver.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/fraction-base.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/e.cpp
    ${PROJECT_SOURCE_DIR}/src/fractions-physics/electric-potential.cpp
	${PROJECT_SOURCE_DIR}/src/full-models/model-one.cpp
    ${PROJECT_SOURCE_DIR}/src/time-iter.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh-refiner.cpp
)

set(LIB_HPP
    ${PROJECT_SOURCE_DIR}/stmod/i-output-provider.hpp
    ${PROJECT_SOURCE_DIR}/stmod/i-variables-storage.hpp
    ${PROJECT_SOURCE_DIR}/stmod/i-steppable.hpp
    ${PROJECT_SOURCE_DIR}/stmod/i-mesh-based.hpp
    ${PROJECT_SOURCE_DIR}/stmod/utils.hpp
    ${PROJECT_SOURCE_DIR}/stmod/poisson-solver.hpp
    ${PROJECT_SOURCE_DIR}/stmod/poisson-solver-adaptor.hpp
    ${PROJECT_SOURCE_DIR}/stmod/poisson-grid.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fe-sampler.hpp
    ${PROJECT_SOURCE_DIR}/stmod/field-output.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions.hpp
    ${PROJECT_SOURCE_DIR}/stmod/matgen.hpp
    ${PROJECT_SOURCE_DIR}/stmod/tensors.hpp
    ${PROJECT_SOURCE_DIR}/stmod/mesh.hpp
    ${PROJECT_SOURCE_DIR}/stmod/mesh-output.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fe-common.hpp
    ${PROJECT_SOURCE_DIR}/stmod/lin-eq-solver.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/fraction-base.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/e.hpp
    ${PROJECT_SOURCE_DIR}/stmod/fractions-physics/electric-potential.hpp
    ${PROJECT_SOURCE_DIR}/stmod/full-models/model-one.hpp
	${PROJECT_SOURCE_DIR}/stmod/phys-consts.hpp
    ${PROJECT_SOURCE_DIR}/stmod/time-iter.hpp
    ${PROJECT_SOURCE_DIR}/stmod/mesh-refiner.hpp
)

add_library(${PROJECT_NAME} STATIC ${LIB_SOURCE} ${LIB_HPP})
target_link_libraries(${PROJECT_NAME} PUBLIC profiler dsiterpp ${DEAL_II_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${DEAL_II_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_rvalue_references)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_lambdas)
