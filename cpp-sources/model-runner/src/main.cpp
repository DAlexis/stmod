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

int main()
{
    AreaConfig config;
    PoissonGrid poisson_grid(config);
    poisson_grid.make_grid();
    PoissonSolver poisson_solver(poisson_grid.triangulation());
    poisson_solver.solve();
    poisson_solver.output("solution-2d.vtk");

    return 0;
}
