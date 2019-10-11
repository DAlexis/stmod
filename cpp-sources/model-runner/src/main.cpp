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
#include "include/fe-sampler.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <cmath>
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
    poisson_solver.output("solution-2d-1.vtk");

    FESampler sampler(poisson_solver.dof_handler());

    auto sol = poisson_solver.solution();
    sampler.sample(sol);
    auto vals = sampler.values();
    for (size_t i=0; i < vals.size(); i++)
    {
        cout << "lap[" << i << "] = " << sampler.laplacians()[i] << endl;
        if (fabs(vals[i] - sol[i]) > 1e-4)
            cout << "vals[i] != sol[i] at i = " << i << " where vals[i] = " << vals[i]  << "and sol[i] = " << sol[i] << endl;
    }

    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(poisson_solver.dof_handler());
    //dealii::Vector<Point<2>> v(sampler.gradients());


    std::vector<std::string> solution_names;
    solution_names.push_back ("GradX");
    solution_names.push_back ("GradY");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation (2, DataComponentInterpretation::component_is_part_of_vector);
    dealii::Vector<double> tmp(sampler.gradients().size() * 2);
    for (size_t i = 0; i<sampler.gradients().size(); i++)
    {
        tmp[2*i] = sampler.gradients()[i][0];
        tmp[2*i + 1] = sampler.gradients()[i][1];
    }

    data_out.add_data_vector(tmp, solution_names, DataOut<2>::type_dof_data, data_component_interpretation); //(sampler.gradients(), "gradients_calculated");
    data_out.build_patches();
    std::ofstream output("gradients.vtk");
    data_out.write_vtk(output);

/*
    poisson_solver.refine_grid();
    poisson_solver.solve();
    poisson_solver.output("solution-2d-2.vtk");


    poisson_solver.refine_grid();
    poisson_solver.solve();
    poisson_solver.output("solution-2d-3.vtk");
*/
    return 0;
}
