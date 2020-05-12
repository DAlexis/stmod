#include "stmod/full-models/model-one.hpp"

#include "stmod/fe-sampler.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions/fraction.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/output/output.hpp"
#include "stmod/time/time-iteration.hpp"
#include "stmod/grid/mesh-refiner.hpp"
#include "stmod/phys-consts.hpp"

#include "stmod/fe-common.hpp"
#include "stmod/grid/grid.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <sstream>

ModelOne::ModelOne()
{
}

ModelOne::~ModelOne()
{
}

void ModelOne::init_grid()
{
    m_grid.load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    //grid.debug_make_rectangular();
    m_boundary_assigner.reset(new BoundaryAssigner(m_grid));
    m_boundary_assigner->assign_boundary_ids();

    m_global_resources.reset(new FEGlobalResources(m_grid.triangulation(), 2));
    m_global_resources->on_triangulation_updated();

    m_refiner.reset(new MeshRefiner(*m_global_resources));

    m_variables_collector.reset(new VariablesCollector(m_global_resources->constraints()));
}

void ModelOne::init_fractions()
{
    m_electric_potential.reset(new ElectricPotential (*m_global_resources));
    m_electric_potential->init_mesh_dependent(m_global_resources->dof_handler());

    m_electrons.reset(new Electrons(*m_global_resources));
    m_electrons->init_mesh_dependent(m_global_resources->dof_handler());

    m_electric_potential->add_charge(m_electrons->values(), - Consts::e);
    m_electrons->set_potential(m_electric_potential->values());

    m_refiner->add_mesh_based(m_electrons.get());
    m_refiner->add_mesh_based(m_electric_potential.get());

    m_output_maker.add(m_electric_potential.get());
    m_output_maker.add(m_electrons.get());

    m_variables_collector->add_pre_step_computator(m_electric_potential.get());
    //m_variables_collector->add_derivatives_provider(m_electrons.get());
    m_variables_collector->add_implicit_steppable(m_electrons.get());
}

void ModelOne::assign_initial_values()
{
    FieldAssigner fa(m_global_resources->dof_handler());
    fa.assign_fiend(
        m_electrons->value_w(),
        [](const dealii::Point<2>& point) -> double
        {
            return 15e13*exp(- (pow((point[0]-0.0075) / 0.002, 2.0) + pow((point[1] - 0.012 ) / 0.005, 2.0)));
        }
    );
}

void ModelOne::run()
{
    m_electric_potential->compute(0.0);
    //m_electrons->solve();

    StmodTimeStepper stepper;
    stepper.init();

    m_output_maker.output(m_global_resources->dof_handler(), "frac-out-iter-" + std::to_string(0) + ".vtu");
    m_refiner->do_refine();
    m_refiner->do_refine();
    //boundary_assigner.assign_boundary_ids();
    m_electric_potential->compute(0.0);
    m_electric_potential->compute(0.0);
    m_output_maker.output(m_global_resources->dof_handler(), "frac-out-iter-" + std::to_string(1) + ".vtu");

    //return 0.0;

    double t = 0;
    double dt = 1e-10;
    double last_output_t = t;
    for (int i = 0; i < 100000; i++)
    {
        /*m_electrons->compute(0.0);
        m_electrons->value_w().add(0.00000005, m_electrons->derivative());
        */
        t = stepper.iterate(*m_variables_collector, t, dt);
/*
        m_boundary_assigner->assign_boundary_ids();
        m_electric_potential->compute(0.0);
        std::cout << "Iteration " << i << "; t = " << t << "... " << std::flush;
        auto & dn = m_electrons->get_implicit_delta(dt, 0.5);

        m_electrons->values_w().add(1.0, dn);
        remove_negative(m_electrons->values_w());
        m_global_resources->constraints().distribute(m_electrons->values_w());*/
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
            m_output_maker.output(m_global_resources->dof_handler(), filename);
            last_output_t = t;
        }
        if (i % 10 == 0)
            m_refiner->do_refine();
    }
}
