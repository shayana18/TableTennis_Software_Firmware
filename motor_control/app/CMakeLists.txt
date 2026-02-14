# Application layer library
file(GLOB APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

if(APP_SOURCES)
    add_library(app_lib STATIC)
    target_sources(app_lib PRIVATE ${APP_SOURCES})
    target_include_directories(app_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(app_lib PUBLIC
        io_lib
        hw_lib
    )
else()
    add_library(app_lib INTERFACE)
    target_include_directories(app_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(app_lib INTERFACE
        io_lib
        hw_lib
    )
endif()
