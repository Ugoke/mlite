cmake_minimum_required(VERSION 3.18)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectMetadata)
mlite_read_project_metadata("${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")

project(
    mlite
    VERSION "${MLITE_PROJECT_VERSION}"
    DESCRIPTION "${MLITE_PROJECT_DESCRIPTION}"
    LANGUAGES CXX
)

option(MLITE_BUILD_PYTHON "Build the Python extension" ON)
option(MLITE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MLITE_INSTALL_CXX "Install the mlite C++ SDK" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CompilerOptions)
include(Sanitizers)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(MLITE_BUILD_TESTS_DEFAULT "${BUILD_TESTING}")
else()
    set(MLITE_BUILD_TESTS_DEFAULT OFF)
endif()
option(MLITE_BUILD_TESTS "Build native mlite tests" ${MLITE_BUILD_TESTS_DEFAULT})
unset(MLITE_BUILD_TESTS_DEFAULT)

find_package(Threads REQUIRED)

add_subdirectory(cpp/foundation)
add_subdirectory(cpp/tensor)
add_subdirectory(cpp/execution)
add_subdirectory(cpp/backends/cpu)
add_subdirectory(cpp/ops)

target_link_libraries(mlite_backend_cpu PUBLIC Threads::Threads)

add_library(mlite INTERFACE)
add_library(mlite::mlite ALIAS mlite)
set_target_properties(mlite PROPERTIES EXPORT_NAME mlite)
target_link_libraries(mlite INTERFACE mlite_ops mlite_backend_cpu)
target_compile_features(mlite INTERFACE cxx_std_20)

mlite_configure_target(mlite_foundation)
mlite_configure_target(mlite_tensor)
mlite_configure_target(mlite_backend_cpu)
mlite_configure_target(mlite_ops)

if(MLITE_ENABLE_SANITIZERS)
    mlite_enable_sanitizers(mlite_foundation)
    mlite_enable_sanitizers(mlite_tensor)
    mlite_enable_sanitizers(mlite_backend_cpu)
    mlite_enable_sanitizers(mlite_ops)
endif()

if(MLITE_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(bindings/python)

    if(MLITE_ENABLE_SANITIZERS)
        # Static-library link options do not reliably propagate to a module.
        # Instrument the final extension explicitly so the sanitizer runtimes
        # are present when the instrumented core libraries are linked.
        mlite_enable_sanitizers(_core)
    endif()
endif()

if(MLITE_BUILD_TESTS)
    enable_testing()
    file(
        GLOB
        MLITE_NATIVE_TEST_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/native/*.cpp"
    )

    foreach(test_source IN LISTS MLITE_NATIVE_TEST_SOURCES)
        get_filename_component(test_name "${test_source}" NAME_WE)
        set(test_target "mlite_test_${test_name}")

        add_executable("${test_target}" "${test_source}")
        target_link_libraries("${test_target}" PRIVATE mlite::mlite)
        target_compile_features("${test_target}" PRIVATE cxx_std_20)
        mlite_configure_target("${test_target}")

        if(MLITE_ENABLE_SANITIZERS)
            mlite_enable_sanitizers("${test_target}")
        endif()

        add_test(NAME "${test_name}" COMMAND "${test_target}")
    endforeach()
endif()

if(MLITE_INSTALL_CXX)
    install(
        TARGETS
            mlite
            mlite_foundation
            mlite_tensor
            mlite_execution
            mlite_backend_cpu
            mlite_ops
        EXPORT mliteTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    foreach(module IN ITEMS foundation tensor execution backends/cpu ops)
        install(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/${module}/include/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT development
            FILES_MATCHING PATTERN "*.hpp"
        )
    endforeach()

    set(MLITE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/mlite")
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mliteConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/mliteConfig.cmake"
        INSTALL_DESTINATION "${MLITE_INSTALL_CMAKEDIR}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/mliteConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion
    )

    install(
        EXPORT mliteTargets
        FILE mliteTargets.cmake
        NAMESPACE mlite::
        DESTINATION "${MLITE_INSTALL_CMAKEDIR}"
        COMPONENT development
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/mliteConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/mliteConfigVersion.cmake"
        DESTINATION "${MLITE_INSTALL_CMAKEDIR}"
        COMPONENT development
    )
endif()
