#ifndef __FE_SAMPLER_HPP__
#define __FE_SAMPLER_HPP__

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>

#include <vector>
#include <functional>

/**
 * @brief The FESampler class produce calculation of coordinates, gradients and laplacians
 * at __support points__ for given finite element solution
 *
 */
class FESampler
{
public:
    enum class Targets : unsigned int
    {
        points = 0,
        values = 1,
        grad_lap = 2,
        all = 3
    };

    FESampler(const dealii::DoFHandler<2>& dof_handler);

    void sample_points();
    void sample(dealii::Vector<double> solution, Targets targets = Targets::all);

    const std::vector<dealii::Point<2>>& points();
    const std::vector<double>&           values();
    const std::vector<dealii::Point<2>>& gradients();
    const std::vector<double>&           laplacians();

private:
    void init_vectors();
    void generate_points();

    const dealii::DoFHandler<2>& m_dof_handler;
    const dealii::FiniteElement<2, 2> &m_fe;
    const std::vector<dealii::Point<2>> &m_support_points;
    const dealii::Quadrature<2> m_quad; // Can be used like i.e. QGauss. It is a storage for points

    std::vector<dealii::Point<2>> m_points;
    std::vector<double> m_values;
    std::vector<dealii::Point<2>> m_gradients;
    std::vector<double> m_laplacians;

    Targets m_last_targets;
};

class FieldAssigner
{
public:
    using FieldFunction = std::function<double (const dealii::Point<2>& point)>;
    FieldAssigner(const dealii::DoFHandler<2>& dof_handler);

    void assign_fiend(dealii::Vector<double>& target_vector, FieldFunction function);

private:
    FESampler m_sampler;

};

#endif // __FE_SAMPLER_HPP__
