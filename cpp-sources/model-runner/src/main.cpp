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

#include "stmod/poisson-solver.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/mesh-output.hpp"
#include "stmod/time-iter.hpp"
#include "stmod/mesh-refiner.hpp"
#include "stmod/phys-consts.hpp"

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
    //grid.debug_make_rectangular();
    BoundaryAssigner boundary_assigner(grid);
    boundary_assigner.assign_boundary_ids();

    FEGlobalResources global_resources(grid.triangulation(), 2);

    global_resources.on_triangulation_updated();


    ElectricPotential pot(global_resources);

    pot.init_mesh_dependent(global_resources.dof_handler());

    Electrons elec(global_resources);
    elec.init_mesh_dependent(global_resources.dof_handler());

    pot.add_charge(elec.values_w(), - Consts::e);
    elec.set_potential(pot.values_w());

    FieldAssigner fa(global_resources.dof_handler());
    fa.assign_fiend(
        elec.value_w(),
        [](const dealii::Point<2>& point) -> double
        {
            return 15e13*exp(- (pow((point[0]-0.0075) / 0.002, 2.0) + pow((point[1] - 0.012 ) / 0.005, 2.0)));
        }
    );

    MeshRefiner refiner(global_resources);
    refiner.add_mesh_based(&elec);
    refiner.add_mesh_based(&pot);


    FractionsOutputMaker output_maker;
    output_maker.add(&pot);
    output_maker.add(&elec);

    pot.compute(0.0);
    //elec.solve();
/*
    VariablesCollector var_coll(global_resources.constraints());
    var_coll.add_pre_step(&pot);
    var_coll.add_steppable(&elec);
*/
    StmodTimeStepper stepper;
    stepper.init();

    output_maker.output(global_resources.dof_handler(), "frac-out-iter-" + std::to_string(0) + ".vtu");
    refiner.do_refine();
    refiner.do_refine();
    //boundary_assigner.assign_boundary_ids();
    pot.compute(0.0);
    pot.compute(0.0);
    output_maker.output(global_resources.dof_handler(), "frac-out-iter-" + std::to_string(1) + ".vtu");

    //return 0.0;

    double t = 0;
    double dt = 1e-10;
    double last_output_t = t;
    for (int i = 0; i < 100000; i++)
    {
        /*elec.compute(0.0);
        elec.value_w().add(0.00000005, elec.derivative());
        */
        boundary_assigner.assign_boundary_ids();
        pot.compute(0.0);
        std::cout << "Iteration " << i << "; t = " << t << "... " << std::flush;
        auto & dn = elec.get_implicit_delta(dt, 0.5);

        elec.values_w().add(1.0, dn);
        remove_negative(elec.values_w());
        global_resources.constraints().distribute(elec.values_w());
        //double dt = stepper.iterate(var_coll, t, 5e-9);
/*
        double t_new = stepper.iterate(var_coll, t, dt);
        dt = t_new - t;
        t = t_new;
        std::cout << "dt = " << dt << std::endl;
*/
        //if (t - last_output_t >= 1e-8)
        {
            std::string filename = "frac-out-iter-" + std::to_string(i+2) + ".vtu";
            std::cout << "Writing " << filename << std::endl;
            output_maker.output(global_resources.dof_handler(), filename);
            last_output_t = t;
        }
        if (i % 10 == 0)
            refiner.do_refine();
    }

    return 0;
}
