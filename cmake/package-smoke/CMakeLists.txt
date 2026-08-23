cmake_minimum_required(VERSION 3.18)

project(mlite_package_smoke LANGUAGES CXX)

find_package(mlite 0.2 CONFIG REQUIRED)

add_executable(mlite_package_smoke main.cpp)
target_link_libraries(mlite_package_smoke PRIVATE mlite::mlite)

enable_testing()
add_test(NAME mlite_package_smoke COMMAND mlite_package_smoke)
