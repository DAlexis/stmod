#include "include/fe-sampler.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

FESampler::FESampler(const dealii::DoFHandler<2>& dof_handler) :
    m_dof_handler(dof_handler)
{
}

void FESampler::sample(dealii::Vector<double> solution)
{
    const auto& fe = m_dof_handler.get_fe();

    // No No No No // USE 'Support poiunts' instead!


    const std::vector< Point< 2 > > &support_points = fe.get_generalized_support_points();

    //Point<2> p(0.0, 0.0); // Point in [0,1]x[0,1] space
    Quadrature<2> quad(support_points); // Can be used like i.e. QGauss. It is a storage for points

    FEValues<2> fe_values(fe, quad,
                          update_values | update_gradients | update_hessians |
                          update_quadrature_points);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quad.size();

    if (dofs_per_cell != n_q_points)
        throw std::runtime_error("Something goes wrong, dofs_per_cell != n_q_points when quadrature points are support points");

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    m_points.resize(solution.size());
    m_values.resize(solution.size());
    m_gradients.resize(solution.size());
    m_laplacians.resize(solution.size());

    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) // dofs_per_cell == number of support points, isn't it??
        {
            const unsigned int current_dof_index = local_dof_indices[i];
            const double solution_component = solution[current_dof_index];
            const auto point = fe_values.quadrature_point(i);
            m_points[current_dof_index] = point;

            // Only one shape func is non-zero on this point, so we do not need iterating along n_q_points
            double shape_value = solution_component * fe_values.shape_value(i, i);
            m_values[current_dof_index] = shape_value;

            // Now iterating over all q_points for current dof (current shape function) and adding components to gradient and laplacian
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
                auto gradient_part = solution_component * fe_values.shape_grad(i, q_index);

                auto hessian = fe_values.shape_hessian_component(i, q_index, 0);
                double laplacian_part = solution_component * (hessian[0][0] + hessian[1][1]);

                // We use local_dof_indices[ >>> q_index <<< ] because q_index is number of support point for q_index's shape sunction in this cell
                m_gradients[local_dof_indices[q_index]] += gradient_part;
                m_laplacians[local_dof_indices[q_index]] += laplacian_part;
            }

        }
        //constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

const std::vector<dealii::Point<2>>& FESampler::points()
{
    return m_points;
}

const std::vector<double>& FESampler::values()
{
    return m_values;
}

const std::vector<dealii::Point<2>>& FESampler::gradients()
{
    return m_gradients;
}

const std::vector<double>& FESampler::laplacians()
{
    return m_laplacians;
}
