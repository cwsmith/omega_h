add_executable(osh_elmToVtx main.cpp)
target_link_libraries(osh_elmToVtx PRIVATE omega_h)
add_test(NAME osh_elmToVtx COMMAND $<TARGET_FILE:osh_elmToVtx>)

add_custom_command(OUTPUT square.msh
    COMMAND ${GMSH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/square.geo
    -2 -o square.msh
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/square.geo
)
add_custom_target(osh_elmToVtx_mesh ALL DEPENDS square.msh)
