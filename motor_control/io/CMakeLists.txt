# IO layer library
add_library(io_lib STATIC)

# Add your IO source files
target_sources(io_lib PRIVATE
    # io/io_module.c
)

# Add include paths for this module
target_include_directories(io_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against hardware layer
target_link_libraries(io_lib PUBLIC
    hw_lib
)
