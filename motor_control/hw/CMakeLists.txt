# Hardware abstraction layer library
add_library(hw_lib STATIC)

# Add your hardware source files
target_sources(hw_lib PRIVATE
    # hw/hw_module.c
)

# Add include paths for this module
target_include_directories(hw_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against STM32 HAL
target_link_libraries(hw_lib PUBLIC
    stm32cubemx
)
