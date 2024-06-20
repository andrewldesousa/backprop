enable_testing()
find_package(GTest REQUIRED)

add_executable(
  test_main
  test_main.cpp
)

target_link_libraries(
  test_main
  GTest::gtest_main
)

target_include_directories(
    test_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(GoogleTest)
gtest_discover_tests(test_main)