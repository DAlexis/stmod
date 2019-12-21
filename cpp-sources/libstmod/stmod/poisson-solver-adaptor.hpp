#ifndef POISSON_SOLVER_ADAPTOR_INCLUDED
#define POISSON_SOLVER_ADAPTOR_INCLUDED

#include "dsiterpp/integration.hpp"

class PoissonSolver;
class FractionsStorage;

class PoissonSolverRHSAdaptor : public dsiterpp::IRHS
{
public:
    PoissonSolverRHSAdaptor(PoissonSolver& solver, FractionsStorage& fractions_storage);

    void pre_iteration_job(double time) override;
    void pre_sub_iteration_job(double time) override;

private:
    PoissonSolver& m_solver;
    FractionsStorage& m_fractions_storage;
};

#endif // POISSON_SOLVER_ADAPTOR_INCLUDED
