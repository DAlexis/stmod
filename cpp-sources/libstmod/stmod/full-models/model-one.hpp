#ifndef FULL_MODELS_MODEL_ONE_HPP_INCLUDED
#define FULL_MODELS_MODEL_ONE_HPP_INCLUDED

#include "stmod/fe-sampler.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"

class Grid;
class PoissonSolver;

class ModelOne
{
public:
    ModelOne();

    void run();

private:
    void init_grid();
    void init_poisson_solver();

    std::shared_ptr<Grid> m_grid;
    std::shared_ptr<PoissonSolver> m_poisson_solver;
};

#endif // FULL_MODELS_MODEL_ONE_HPP_INCLUDED
