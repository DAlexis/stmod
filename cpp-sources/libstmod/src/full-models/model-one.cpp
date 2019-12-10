#include "stmod/full-models/model-one.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/poisson-grid.hpp"
#include "dsiterpp/time-iter.hpp"
#include "dsiterpp/runge-error-estimator.hpp"
#include "dsiterpp/runge-kutta.hpp"

using namespace dsiterpp;

ModelOne::ModelOne()
{
    init_grid();
    init_poisson_solver();
    init_fractions_storage();
    init_electrons();
}

void ModelOne::run()
{
}

void ModelOne::output_potential(const std::string& filename)
{
    m_poisson_solver->output(filename);
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
}

void ModelOne::init_fractions_storage()
{
    m_frac_storage = std::make_shared<FractionsStorage>();
    m_frac_storage->create_arrays(size_t(Fractions::count), m_poisson_solver->dof_handler().n_dofs());
}

void ModelOne::init_electrons()
{
    m_electrons_rhs = std::make_shared<ElectronsRHS>(*m_frac_storage, size_t(Fractions::electrons), m_poisson_solver->solution(), m_poisson_solver->dof_handler());
}

void ModelOne::init_time_iterator()
{
    m_time_iterator = std::make_shared<TimeIterator>();

    //m_time_iterator->set_continious_iterator
}
