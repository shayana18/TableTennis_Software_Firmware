# IO layer library
file(GLOB IO_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

if(IO_SOURCES)
    add_library(io_lib STATIC)
    target_sources(io_lib PRIVATE ${IO_SOURCES})
    target_include_directories(io_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(io_lib PUBLIC
        hw_lib
        datatypes_lib
    )
else()
    add_library(io_lib INTERFACE)
    target_include_directories(io_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(io_lib INTERFACE
        hw_lib
        datatypes_lib
    )
endif()
