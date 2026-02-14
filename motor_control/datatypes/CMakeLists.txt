message(" ")
message("Adding datatypes lib")

add_library(datatypes_lib)

target_sources(datatypes_lib PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/ringBuffer.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/mailbox.c")

target_include_directories(datatypes_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
