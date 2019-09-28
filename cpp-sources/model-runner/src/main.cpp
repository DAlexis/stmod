/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */

#include "include/poisson-grid.hpp"
#include "include/poisson-solver.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>

using namespace dealii;
using namespace std;

int main()
{
    AreaConfig config;
    PoissonGrid poisson_grid(config);

    Triangulation<2> triangulation;
    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh");
    gridin.read_msh(f);

    cout << "Mesh reading done" << endl;

    //poisson_grid.make_grid();
    //PoissonSolver poisson_solver(poisson_grid.triangulation());
    BoundaryAssigner::assign_boundary_ids(triangulation);
    PoissonSolver poisson_solver(triangulation);
    poisson_solver.solve();
    poisson_solver.output("solution-2d.vtk");

    return 0;
}
