cmake_minimum_required(VERSION 2.8)

#set (CMAKE_CXX_STANDARD 14)

find_package(deal.II REQUIRED)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(stmod-runner)

find_package (Boost COMPONENTS date_time program_options REQUIRED)
find_package (Threads)


set(EXE_SOURCES
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${EXE_SOURCES})
DEAL_II_SETUP_TARGET(${PROJECT_NAME})

target_link_libraries (${PROJECT_NAME} stmod)
