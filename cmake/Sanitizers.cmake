function(mlite_enable_sanitizers target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "Cannot enable sanitizers for unknown target '${target_name}'")
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /fsanitize=address)
        target_link_options(${target_name} PRIVATE /fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            ${target_name}
            PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
    else()
        message(
            WARNING
            "MLITE_ENABLE_SANITIZERS is not supported by ${CMAKE_CXX_COMPILER_ID}; "
            "target '${target_name}' will not be instrumented"
        )
    endif()
endfunction()
