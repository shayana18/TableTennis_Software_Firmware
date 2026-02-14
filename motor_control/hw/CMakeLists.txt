# Hardware abstraction layer library
file(GLOB HW_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

if(HW_SOURCES)
    add_library(hw_lib STATIC)
    target_sources(hw_lib PRIVATE ${HW_SOURCES})
    target_include_directories(hw_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(hw_lib PUBLIC
        stm32cubemx
    )
else()
    add_library(hw_lib INTERFACE)
    target_include_directories(hw_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(hw_lib INTERFACE
        stm32cubemx
    )
endif()
