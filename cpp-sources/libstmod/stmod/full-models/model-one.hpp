#ifndef FULL_MODELS_MODEL_ONE_HPP_INCLUDED
#define FULL_MODELS_MODEL_ONE_HPP_INCLUDED

#include "dsiterpp/time-iter.hpp"
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

    void output_hook(double real_time);

private:
    void init_grid();
    void init_poisson_solver();
    void init_fractions_storage();
    void init_electrons();

    void init_time_iterator();

    std::shared_ptr<Grid> m_grid;
};

#endif // FULL_MODELS_MODEL_ONE_HPP_INCLUDED
