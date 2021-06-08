#include "stmod/utils/symmetry-axis.hpp"

#include <deal.II/fe/fe_values.h>

#include <iostream>

using namespace dealii;

AxisRegularizer::AxisRegularizer(const Grid& grid) :
    m_grid(grid)
{

}

void AxisRegularizer::regularize(dealii::Vector<double>& values, const FEGlobalResources& fe_res)
{
    const double epsilon = 1e-8;
    const dealii::Quadrature<2>& quadrature = fe_res.fe().get_generalized_support_points();
    const FiniteElement<2, 2>& fe = fe_res.fe();

    FEValues<2, 2> fe_values(fe, quadrature, update_quadrature_points);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : fe_res.dof_handler().active_cell_iterators())
    {
        if (!cell->at_boundary())
            continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        std::vector<types::global_dof_index> points_on_axis;
        std::vector<types::global_dof_index> other_points;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];

            if (r < epsilon)
            {
                // This point is on axis
                points_on_axis.push_back(local_dof_indices[q_index]);
            } else {
                other_points.push_back(local_dof_indices[q_index]);
            }
        }

        if (points_on_axis.empty())
            continue;

        double other_points_average = 0;
        for (auto other_point_index : other_points)
        {
            other_points_average += values[other_point_index];
        }
        other_points_average /= other_points.size();

        for (auto points_on_axis_index : points_on_axis)
        {
            values[points_on_axis_index] = other_points_average;
        }
        /*
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            values[local_dof_indices[q_index]] = other_points_average;
        }*/
    }
}
/*
std::vector<dealii::types::global_dof_index> get_axis_points()
{
}
*/
