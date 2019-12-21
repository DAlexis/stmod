#ifndef FULL_MODELS_MODEL_ONE_HPP_INCLUDED
#define FULL_MODELS_MODEL_ONE_HPP_INCLUDED

#include <memory>

class Grid;
class PoissonSolver;
class FractionsStorage;
class ElectronsRHS;
class PoissonSolverRHSAdaptor;
namespace dsiterpp {
    class TimeIterator;
    class RHSGroup;
    class IIntegrator;
    class IErrorEstimator;
}

class ModelOne
{
public:
    enum class Fractions : unsigned int
    {
        electrons = 0,
        count
    };

    ModelOne();

    void run();
    void output_potential(const std::string& filename);
    void output_fractions(const std::string& filename);

private:
    void init_grid();
    void init_poisson_solver();
    void init_fractions_storage();
    void init_electrons();

    void init_time_iterator();

    std::shared_ptr<Grid> m_grid;
    std::shared_ptr<PoissonSolver> m_poisson_solver;
    std::shared_ptr<FractionsStorage> m_frac_storage;

    std::shared_ptr<PoissonSolverRHSAdaptor> m_poisson_solver_adaptor;

    std::shared_ptr<ElectronsRHS> m_electrons_rhs;

    std::shared_ptr<dsiterpp::TimeIterator> m_time_iterator;
    std::shared_ptr<dsiterpp::RHSGroup> m_RHSs;
    std::shared_ptr<dsiterpp::IErrorEstimator> m_error_estimator;
    std::shared_ptr<dsiterpp::IIntegrator> m_intergator;
};

#endif // FULL_MODELS_MODEL_ONE_HPP_INCLUDED
