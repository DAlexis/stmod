#include "stmod/fe-sampler.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <stdexcept>

using namespace dealii;

FESampler::FESampler(const dealii::DoFHandler<2>& dof_handler) :
    m_dof_handler(dof_handler),
    m_fe(m_dof_handler.get_fe()),
    m_support_points(m_fe.get_generalized_support_points()),
    m_quad(m_support_points)
{
    init_vectors();
    generate_points();
}

void FESampler::init_vectors()
{
    m_points.resize(m_dof_handler.n_dofs());
    m_values.resize(m_dof_handler.n_dofs());
    m_gradients.resize(m_dof_handler.n_dofs());
    m_laplacians.resize(m_dof_handler.n_dofs());
}

void FESampler::sample_points()
{
    FEValues<2> fe_values(m_fe, m_quad, update_quadrature_points);

    const unsigned int dofs_per_cell = m_fe.dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) // dofs_per_cell == number of support points, isn't it??
        {
            const unsigned int current_dof_index = local_dof_indices[i];
            m_points[current_dof_index] = fe_values.quadrature_point(i);
        }
    }
}

void FESampler::sample(dealii::Vector<double> solution, FESampler::Targets targets)
{
    if (m_values.size() != m_dof_handler.n_dofs())
    {
        init_vectors();
    }

    if (m_dof_handler.n_dofs() != solution.size())
    {
        throw std::range_error("FESampler::sample: m_dof_handler.n_dofs() != solution.size()");
    }

    FEValues<2> fe_values(m_fe, m_quad, update_values |
                          update_gradients | update_hessians |
                          update_quadrature_points);

    const unsigned int dofs_per_cell = m_fe.dofs_per_cell;
    const unsigned int n_q_points    = m_quad.size();

    if (dofs_per_cell != n_q_points)
        throw std::runtime_error("Something goes wrong, dofs_per_cell != n_q_points when quadrature points are support points");

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);


        for (unsigned int i = 0; i < dofs_per_cell; ++i) // dofs_per_cell == number of support points, isn't it??
        {
            const unsigned int current_dof_index = local_dof_indices[i];
            m_points[current_dof_index] = fe_values.quadrature_point(i);

            if (! (static_cast<unsigned int>(targets) & (static_cast<unsigned int>(Targets::values) | static_cast<unsigned int>(Targets::grad_lap))))
                continue;

            const double solution_component = solution[current_dof_index];

            // Only one shape func is non-zero on this point, so we do not need iterating along n_q_points
            double shape_value = solution_component * fe_values.shape_value(i, i);
            m_values[current_dof_index] = shape_value;


            if (! (static_cast<unsigned int>(targets) & static_cast<unsigned int>(Targets::grad_lap)) )
                continue;

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
    }
    m_last_targets = targets;
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

void FESampler::generate_points()
{
    FEValues<2> fe_values(m_fe, m_quad, update_quadrature_points);

    const unsigned int dofs_per_cell = m_fe.dofs_per_cell;
    const unsigned int n_q_points    = m_quad.size();

    if (dofs_per_cell != n_q_points)
        throw std::runtime_error("Something goes wrong, dofs_per_cell != n_q_points when quadrature points are support points");

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int current_dof_index = local_dof_indices[i];
            const auto point = fe_values.quadrature_point(i);
            m_points[current_dof_index] = point;
        }
    }
}

FieldAssigner::FieldAssigner(const dealii::DoFHandler<2>& dof_handler) :
    m_sampler(dof_handler)
{
}

void FieldAssigner::assign_fiend(dealii::Vector<double>& target_vector, FieldFunction function)
{
    m_sampler.sample_points();
    for (unsigned int i = 0; i < target_vector.size(); i++)
    {
        target_vector[i] = function(m_sampler.points()[i]);
    }
}
