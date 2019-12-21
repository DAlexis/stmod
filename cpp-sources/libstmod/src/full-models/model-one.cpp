#include "stmod/full-models/model-one.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/poisson-grid.hpp"
#include "stmod/poisson-solver-adaptor.hpp"

#include "dsiterpp/time-iter.hpp"
#include "dsiterpp/runge-error-estimator.hpp"
#include "dsiterpp/runge-kutta.hpp"
#include "stmod/field-output.hpp"

#include <fstream>

using namespace dsiterpp;

ModelOne::ModelOne()
{
    init_grid();
    init_poisson_solver();
    init_fractions_storage();
    init_electrons();
    init_time_iterator();
}

void ModelOne::run()
{
    m_time_iterator->iterate();
    /*m_time_iterator->iterate();
    m_time_iterator->iterate();
    m_time_iterator->iterate();*/
}

void ModelOne::output_potential(const std::string& filename)
{
    m_poisson_solver->output(filename);
}

void ModelOne::output_fractions(const std::string& filename)
{
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(m_poisson_solver->dof_handler());
    data_out.add_data_vector(m_frac_storage->previous(0), "electrons");

    data_out.build_patches();
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

void ModelOne::init_grid()
{
    m_grid = std::make_shared<Grid>();
    m_grid->load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    BoundaryAssigner boundary_assigner(*m_grid);
    boundary_assigner.assign_boundary_ids();
}

void ModelOne::init_poisson_solver()
{
    m_poisson_solver = std::make_shared<PoissonSolver>(m_grid->triangulation());
    m_poisson_solver->solve(); // For initialization
}

void ModelOne::init_fractions_storage()
{
    m_frac_storage = std::make_shared<FractionsStorage>();
    m_frac_storage->create_arrays(size_t(Fractions::count), m_poisson_solver->dof_handler().n_dofs());
    m_RHSs = std::make_shared<RHSGroup>();
}

void ModelOne::init_electrons()
{
    m_electrons_rhs = std::make_shared<ElectronsRHS>(*m_frac_storage, size_t(Fractions::electrons), m_poisson_solver->solution(), m_poisson_solver->dof_handler());
    m_RHSs->add_rhs(m_electrons_rhs.get());
    /*FieldAssigner fa(m_poisson_solver->dof_handler());
    fa.assign_fiend(
        m_frac_storage->previous_w(0),
        [](const dealii::Point<2>& point) -> double
        {
            return point[0];
        }
    );*/

}

void ModelOne::init_time_iterator()
{
    m_poisson_solver_adaptor = std::make_shared<PoissonSolverRHSAdaptor>(*m_poisson_solver, *m_frac_storage);
    m_RHSs->add_rhs(m_poisson_solver_adaptor.get());

    m_intergator = std::make_shared<RungeKuttaIterator>();
    m_error_estimator = std::make_shared<RungeErrorEstimator>();

    m_time_iterator = std::make_shared<TimeIterator>();

    m_time_iterator->set_rhs(m_RHSs.get());
    m_time_iterator->set_variable(m_frac_storage.get());
    m_time_iterator->set_continious_iterator(m_intergator.get());
    //m_time_iterator->set_error_estimator(m_error_estimator.get());
}
