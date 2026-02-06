# Application layer library
add_library(app_lib STATIC)

# Add your application source files
target_sources(app_lib PRIVATE
    # app/app_module.c
)

# Add include paths for this module
target_include_directories(app_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against IO and hardware layers
target_link_libraries(app_lib PUBLIC
    io_lib
    hw_lib
)
