project(stmod-unit-tests)

set(EXE_SOURCES
    ${PROJECT_SOURCE_DIR}/test-tensors.cpp
    ${PROJECT_SOURCE_DIR}/test-utils.cpp
)

set(EXE_HPP

)

add_executable(${PROJECT_NAME} ${EXE_SOURCES} ${EXE_HPP})

target_link_libraries (${PROJECT_NAME}
    gtest
    gtest_main
    stmod
)

add_test(NAME stmod-testing COMMAND ${PROJECT_NAME})
