#include "stmod/matgen.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;


void create_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints,
        const dealii::Quadrature<2> & quadrature)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            //const auto x_q = fe_values.quadrature_point(q_index);
            //const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += (
                        fe_values.shape_grad(i, q_index) * // grad phi_i
                        fe_values.shape_grad(j, q_index) * // grad phi_j
                        //r *                                // r
                        fe_values.JxW(q_index)             // dx
                    );
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void create_r_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints,
        const dealii::Quadrature<2> & quadrature,
        double r_epsilon)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values | update_jacobians);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            const double z = x_q[1];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    auto grad_psi_i = fe_values.shape_grad(i, q_index);
                    auto grad_psi_j = fe_values.shape_grad(j, q_index);

                    //double psi_i = fe_values.shape_value(i, q_index);
                    double psi_j = fe_values.shape_value(j, q_index);

                    cell_matrix(i, j) +=
                        (
                            - grad_psi_i * grad_psi_j * (r + r_epsilon)
                        ) * fe_values.JxW(q_index);

                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void create_r_mass_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints,
        const dealii::Quadrature<2> & quadrature,
        double r_epsilon)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    double psi_i = fe_values.shape_value(i, q_index);
                    double psi_j = fe_values.shape_value(j, q_index);

                    cell_matrix(i, j) +=
                        (
                            psi_i * (r + r_epsilon) * psi_j
                        ) * fe_values.JxW(q_index);
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void create_phi_i_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature,
        double r_epsilon)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    FullTensor3 cell_tensor(dofs_per_cell, dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_tensor = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        cell_tensor(i, j, k) += (
                            fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                            fe_values.shape_value(j, q_index) *   // phi_j(x_q)
                            fe_values.shape_value(k, q_index) *   // phi_k(x_q)
                            (r + r_epsilon) *                     // r
                            fe_values.JxW(q_index)                // dx
                        );
                    }
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    unsigned int global_i = local_dof_indices[i];
                    unsigned int global_j = local_dof_indices[j];
                    unsigned int global_k = local_dof_indices[k];
                    tensor.set(global_i, global_j, global_k, tensor(global_i, global_j, global_k) + cell_tensor(i, j, k));
                }
            }

        }
    }
}

void create_grad_phi_i_grad_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature,
        double r_epsilon)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    FullTensor3 cell_tensor(dofs_per_cell, dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_tensor = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        cell_tensor(i, j, k) += (
                            fe_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                            fe_values.shape_grad(j, q_index) *   // grad phi_j(x_q)
                            fe_values.shape_value(k, q_index) *   // phi_k(x_q)
                            (r + r_epsilon) *                     // r
                            fe_values.JxW(q_index)                // dx
                        );
                    }
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    unsigned int global_i = local_dof_indices[i];
                    unsigned int global_j = local_dof_indices[j];
                    unsigned int global_k = local_dof_indices[k];
                    tensor.set(global_i, global_j, global_k, tensor(global_i, global_j, global_k) + cell_tensor(i, j, k));
                }
            }

        }
    }
}

void sum_with_tensor(dealii::Vector<double>& out_vector,
                     const dealii::Vector<double>& in_first,
                     const dealii::Vector<double>& in_second,
                     const SparseTensor3& tensor)
{
    for (auto & indexes : tensor.nonzero())
    {
        SparseTensor3::IndexType i = std::get<0>(indexes);
        SparseTensor3::IndexType j = std::get<1>(indexes);
        SparseTensor3::IndexType k = std::get<2>(indexes);
        out_vector[k] += tensor(i, j, k) * in_first[i] * in_second[k];
    }
}
