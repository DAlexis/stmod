cmake_minimum_required(VERSION 2.8)

project(stmod-runner)

find_package (Boost COMPONENTS date_time program_options REQUIRED)

set(EXE_SOURCES
    src/main.cpp
)

set(${PROJECT_NAME}_USED_INCDIRS
    ${stmod_INCLUDE_DIRS}
)

include_directories(
    ${${PROJECT_NAME}_USED_INCDIRS}
)

add_executable(${PROJECT_NAME} ${EXE_SOURCES})

target_link_libraries (${PROJECT_NAME}
    ${stmod_LIBRARY}
)

# This is for unit tests
export_include_dirs(
    ${PROJECT_SOURCE_DIR}
    ${${PROJECT_NAME}_USED_INCDIRS}
)
