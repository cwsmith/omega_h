add_executable(osh_laplacian main.cpp)
target_link_libraries(osh_laplacian PRIVATE omega_h)
add_test(NAME osh_laplacian COMMAND $<TARGET_FILE:osh_laplacian>)

add_custom_command(OUTPUT square.msh
    COMMAND ${GMSH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/square.geo
    -2 -o square.msh
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/square.geo
)
add_custom_target(osh_laplacian_mesh ALL DEPENDS square.msh)
