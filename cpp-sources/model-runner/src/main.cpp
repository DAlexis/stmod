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

#include "stmod/poisson-grid.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/mesh-output.hpp"
#include "stmod/time-iter.hpp"

#include "stmod/full-models/model-one.hpp"
#include "stmod/fe-common.hpp"
#include "stmod/mesh.hpp"

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
    Grid grid;
    grid.load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    BoundaryAssigner boundary_assigner(grid);
    boundary_assigner.assign_boundary_ids();

    FEResources fe_res(grid.triangulation(), 1);
    fe_res.init();

    ElectricPotential pot(fe_res);
    pot.init();

    Electrons elec(fe_res);
    elec.init();

    FieldAssigner fa(fe_res.dof_handler());
    fa.assign_fiend(
        elec.value_w(),
        [](const dealii::Point<2>& point) -> double
        {
            return exp(-pow((point[0]-0.008) / 0.002, 2.0));
            //return 0.5;
            //return exp(-point[0] / 0.002 * ());
            /*if (point[0] < 0.001)
                return 1.0;
            else
                return 0.0;*/
            //return point[0];
        }
    );

    FractionsOutputMaker output_maker;
    output_maker.add(&pot);
    output_maker.add(&elec);

    pot.solve();
    //elec.solve();

    VariablesCollector var_coll;
    var_coll.add_steppable(&elec);

    StmodTimeStepper stepper;
    stepper.init();

    double t = 0;
    double dt = 1e-5;
    for (int i = 0; i < 100000; i++)
    {
        /*elec.compute(0.0);
        elec.value_w().add(0.00000005, elec.derivative());
        */
        std::cout << "Iteration " << i << "; t = " << t << "... " << std::flush;
        //double dt = stepper.iterate(var_coll, t, 5e-9);

        double t_new = stepper.iterate(var_coll, t, dt);
        dt = t_new - t;
        t = t_new;
        std::cout << "dt = " << dt << std::endl;

        if (i % 100 == 0)
            output_maker.output(fe_res.dof_handler(), "frac-out-iter-" + std::to_string(i) + ".vtu");
    }



    /*ModelOne model;
    model.output_fractions("fractions.vtk");
    model.run();*/
    //model.output_fractions("fractions.vtk");

    /*

    Grid grid;
    grid.load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );

    BoundaryAssigner boundary_assigner(grid);
    boundary_assigner.assign_boundary_ids();
    //boundary_assigner.assign_manifold_ids();
    PoissonSolver poisson_solver(grid.triangulation());
    poisson_solver.solve();
    poisson_solver.output("solution-2d-1.vtk");

    FractionsStorage frac_storage;
    frac_storage.create_arrays(1, poisson_solver.solution().size());
    ElectronsRHS electrons_rhs(frac_storage, 0, poisson_solver.solution(), poisson_solver.dof_handler());
    electrons_rhs.calculate_rhs(0.0);

    auto sol = poisson_solver.solution();
    std::vector<dealii::Vector<double>> old_sol;
    old_sol.push_back(sol);

    poisson_solver.estimate_error();
    std::vector<dealii::Vector<double>> interpolated_sol;
    auto refined_solution = poisson_solver.refine_and_coarsen_grid(old_sol);



    FESampler sampler(poisson_solver.dof_handler());

    sampler.sample(refined_solution[0]);
    auto vals = sampler.values();
    for (size_t i=0; i < vals.size(); i++)
    {
        cout << "lap[" << i << "] = " << sampler.laplacians()[i] << endl;
    }

    VectorOutputMaker vector_out_maker(poisson_solver.triangulation(), poisson_solver.polynomial_degree());
    vector_out_maker.set_vector(sampler.gradients(), {"Ex", "Ey"});
    vector_out_maker.output("gradients-test.vtk");

    poisson_solver.solve();
    poisson_solver.output("solution-2d-2.vtk");
    */
/*
    poisson_solver.estimate_error();
    poisson_solver.refine_and_coarsen_grid();
    poisson_solver.solve();
    poisson_solver.output("solution-2d-2.vtk");


    poisson_solver.estimate_error();
    poisson_solver.refine_and_coarsen_grid();
    poisson_solver.solve();
    poisson_solver.output("solution-2d-3.vtk");
*/
    return 0;
}
