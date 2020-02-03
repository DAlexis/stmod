#include "stmod/full-models/model-one.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/mesh.hpp"
#include "stmod/poisson-solver-adaptor.hpp"

#include "dsiterpp/time-iter.hpp"
#include "dsiterpp/runge-error-estimator.hpp"
#include "dsiterpp/runge-kutta.hpp"
#include "dsiterpp/euler-explicit.hpp"
#include "stmod/field-output.hpp"

#include <fstream>
#include <sstream>

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
    m_time_iterator->set_stop_time(1.0);
    //m_time_iterator->run();
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
    data_out.add_data_vector(m_frac_storage->previous(static_cast<unsigned int>(Fractions::electrons)), "electrons");
    data_out.add_data_vector(m_frac_storage->previous(static_cast<unsigned int>(Fractions::electrons)), "electrons2");


    FESampler electrons_sampler(m_poisson_solver->dof_handler());
    electrons_sampler.sample(m_frac_storage->previous(static_cast<unsigned int>(Fractions::electrons)));
    dealii::Vector<double> laps(electrons_sampler.laplacians().size());
    for (size_t i = 0; i < electrons_sampler.laplacians().size(); i++)
    {
        laps[i] = electrons_sampler.laplacians()[i];
    }

    if (laps.size() != m_frac_storage->previous(static_cast<unsigned int>(Fractions::electrons)).size())
    {
        throw std::runtime_error("Size is not equal!");
    }

    data_out.add_data_vector(laps, "electrons_laplacians");

    data_out.build_patches();
    std::ofstream output(filename.c_str());
    data_out.write_vtu(output);
}

void ModelOne::output_hook(double real_time)
{
    std::stringstream sstream;
    sstream.setf(std::ios::fixed);
    sstream.precision(10);
    sstream << "t=" << real_time;
    std::string potential_file_name = "potential-" + sstream.str() + ".vtu";
    std::string fractions_file_name = "fractions-" + sstream.str() + ".vtu";
    output_potential(potential_file_name);
    output_fractions(fractions_file_name);
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
    m_poisson_solver = std::make_shared<PoissonSolver>(m_grid->triangulation(), 3);
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
    FieldAssigner fa(m_poisson_solver->dof_handler());
    fa.assign_fiend(
        m_frac_storage->previous_w(0),
        [](const dealii::Point<2>& point) -> double
        {
            return exp(-point[0] / 0.002);
            /*if (point[0] < 0.001)
                return 1.0;
            else
                return 0.0;*/
            //return point[0];
        }
    );

}

void ModelOne::init_time_iterator()
{
    m_poisson_solver_adaptor = std::make_shared<PoissonSolverRHSAdaptor>(*m_poisson_solver, *m_frac_storage);
    m_RHSs->add_rhs(m_poisson_solver_adaptor.get());

    //m_intergator = std::make_shared<RungeKuttaIterator>();
    m_intergator = std::make_shared<EulerExplicitIterator>();
    m_error_estimator = std::make_shared<RungeErrorEstimator>();

    m_time_iterator = std::make_shared<TimeIterator>();

    m_time_iterator->set_rhs(m_RHSs.get());
    m_time_iterator->set_variable(m_frac_storage.get());
    m_time_iterator->set_continious_iterator(m_intergator.get());
    m_time_iterator->set_step(0.00000002);
    //m_time_iterator->set_step(0.00001);
    //m_time_iterator->set_step(0);
    //m_time_iterator->set_error_estimator(m_error_estimator.get());

    m_output_hook = std::make_shared<TimeHookPeriodicFunc>(
        [this](double real_time, double wanted_time)
        {
            DSITERPP_UNUSED(wanted_time);
            output_hook(real_time);
        }
    );
    m_output_hook->set_period(0.0);
    m_time_iterator->add_hook(m_output_hook.get());
}
