#ifndef __FE_SAMPLER_HPP__
#define __FE_SAMPLER_HPP__

#include <deal.II/dofs/dof_handler.h>

#include <vector>

/**
 * @brief The FESampler class produce calculation of coordinates, gradients and laplacians
 * at __support points__ for given finite element solution
 *
 */
class FESampler
{
public:
    FESampler(const dealii::DoFHandler<2>& dof_handler);

    void sample(dealii::Vector<double> solution);

    const std::vector<dealii::Point<2>>& points();
    const std::vector<double>&           values();
    const std::vector<dealii::Point<2>>& gradients();
    const std::vector<double>&           laplacians();

private:
    const dealii::DoFHandler<2>& m_dof_handler;
    std::vector<dealii::Point<2>> m_points;
    std::vector<double> m_values;
    std::vector<dealii::Point<2>> m_gradients;
    std::vector<double> m_laplacians;

};

#endif // __FE_SAMPLER_HPP__
