#include "stmod/poisson-solver-adaptor.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/fractions.hpp"

using namespace dsiterpp;

PoissonSolverRHSAdaptor::PoissonSolverRHSAdaptor(PoissonSolver& solver, FractionsStorage& fractions_storage) :
    m_solver(solver), m_fractions_storage(fractions_storage)
{
}

void PoissonSolverRHSAdaptor::pre_iteration_job(double)
{
    m_solver.estimate_error();
    m_fractions_storage.resize_interpolate(
        [this](const std::vector<dealii::Vector<double>>& orig_fractions_vectors) -> std::vector<dealii::Vector<double>>
        {
            return m_solver.refine_and_coarsen_grid(orig_fractions_vectors);
        }
    );
}

void PoissonSolverRHSAdaptor::pre_sub_iteration_job(double)
{
    m_solver.solve();
}
