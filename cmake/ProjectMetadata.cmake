function(mlite_read_project_metadata pyproject_path)
    if(NOT EXISTS "${pyproject_path}")
        message(FATAL_ERROR "pyproject.toml was not found: ${pyproject_path}")
    endif()

    file(STRINGS "${pyproject_path}" pyproject_lines ENCODING UTF-8)

    set(in_project_section FALSE)
    foreach(line IN LISTS pyproject_lines)
        string(STRIP "${line}" stripped_line)

        if(stripped_line MATCHES "^\\[([^]]+)\\][ \t]*(#.*)?$")
            if("${CMAKE_MATCH_1}" STREQUAL "project")
                set(in_project_section TRUE)
            elseif(in_project_section)
                break()
            else()
                set(in_project_section FALSE)
            endif()
            continue()
        endif()

        if(NOT in_project_section OR stripped_line MATCHES "^(#|$)")
            continue()
        endif()

        foreach(field IN ITEMS name version description)
            if(DEFINED metadata_${field})
                continue()
            endif()

            if(stripped_line MATCHES "^${field}[ \t]*=[ \t]*\"([^\"]*)\"[ \t]*(#.*)?$")
                set(metadata_${field} "${CMAKE_MATCH_1}")
            elseif(stripped_line MATCHES "^${field}[ \t]*=[ \t]*'([^']*)'[ \t]*(#.*)?$")
                set(metadata_${field} "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endforeach()

    foreach(field IN ITEMS name version description)
        if(NOT DEFINED metadata_${field} OR metadata_${field} STREQUAL "")
            message(
                FATAL_ERROR
                "Required [project] metadata '${field}' is missing or empty in ${pyproject_path}"
            )
        endif()

        string(TOUPPER "${field}" upper_field)
        set("MLITE_PROJECT_${upper_field}" "${metadata_${field}}" PARENT_SCOPE)
    endforeach()
endfunction()
