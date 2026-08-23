add_library(
    mlite_backend_cpu
    STATIC
    src/cpu_allocator.cpp
    src/thread_pool_executor.cpp
)
add_library(mlite::backend_cpu ALIAS mlite_backend_cpu)
set_target_properties(mlite_backend_cpu PROPERTIES EXPORT_NAME backend_cpu)

target_include_directories(
    mlite_backend_cpu
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(mlite_backend_cpu PUBLIC mlite_execution)
target_compile_features(mlite_backend_cpu PUBLIC cxx_std_20)
