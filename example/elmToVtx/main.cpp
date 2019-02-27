#include <cmath>
#include <iostream>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"

namespace oh = Omega_h;

int main(int argc, char** argv) {
  auto lib = oh::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = oh::gmsh::read("square.msh", world);
  const auto dim = mesh.dim();
  const auto verts2elems = mesh.ask_up(oh::VERT, dim);
  oh::Write<oh::LO> elmVal(mesh.nelems(),1);
  oh::Write<oh::Real> vtxDensity(mesh.nverts(),0);
  const auto accumulate = OMEGA_H_LAMBDA(oh::LO i) {
    const auto deg = verts2elems.a2ab[i+1]-verts2elems.a2ab[i];
    const auto firstElm = verts2elems.a2ab[i];
    for (int j = 0; j < deg; j++){
      const auto elm = verts2elems.ab2b[firstElm+j];
      vtxDensity[i] += elmVal[elm];
    }
    printf("vtx %d density %.0f\n", i, vtxDensity[i]);
  };
  oh::parallel_for(mesh.nverts(), accumulate, "accumulate");
  mesh.add_tag(oh::VERT, "density", 1, oh::Reals(vtxDensity));
  oh::vtk::write_parallel("rendered", &mesh, dim);
  return 0;
}
