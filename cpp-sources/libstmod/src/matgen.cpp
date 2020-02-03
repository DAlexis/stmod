#include "stmod/matgen.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>

using namespace dealii;

FullTensor::FullTensor(unsigned int size_i, unsigned int size_j, unsigned int size_k) :
    m_size_i(size_i), m_size_j(size_j), m_size_k(size_k)
{
    m_content.resize(m_size_i * m_size_j * m_size_k, 0.0);
}

double& FullTensor::operator()(unsigned int i, unsigned int j, unsigned int k)
{
#ifdef DEBUG
    if (i >= m_size_i || j >= m_size_j || k >= m_size_k)
        throw std::range_error("FullTensor error: index out of range");
#endif
    return m_content[k + j * m_size_k + i * m_size_k * m_size_j];
}

void FullTensor::operator=(double x)
{
    std::fill(m_content.begin(), m_content.end(), x);
}

double SparseTensor::operator()(unsigned int i, unsigned int j, unsigned int k)
{
    auto it = m_unordered_map.find(i);
    if (it == m_unordered_map.end())
        return 0.0;

    auto jt = it->second.find(j);
    if (jt == it->second.end())
        return 0.0;

    auto kt = jt->second.find(j);
    if (kt == jt->second.end())
        return 0.0;

    return kt->second;
}

void SparseTensor::set(unsigned int i, unsigned int j, unsigned int k, double value)
{
    m_unordered_map[i][j][k] = value;
}


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
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
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
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    auto grad_psi_i = fe_values.shape_grad(i, q_index);
                    auto grad_psi_j = fe_values.shape_grad(j, q_index);

                    double psi_i = fe_values.shape_value(i, q_index);
                    double psi_j = fe_values.shape_value(j, q_index);

                    cell_matrix(i, j) +=
                        (
                            grad_psi_i[0] * psi_j +
                            grad_psi_i[0] * (psi_j + r * grad_psi_j[0]) + grad_psi_i[1] * grad_psi_j[1]
                        ) * fe_values.JxW(q_index);
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void create_mass_matrix_axial(
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
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += (
                        fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.shape_value(j, q_index) * // phi_j(x_q)
                        r *                                 // r
                        fe_values.JxW(q_index)              // dx
                    );
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        /*for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            sparse_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
        }*/
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void add_dirichlet_rhs_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::Vector<double>& system_rhs,
        const std::map<dealii::types::boundary_id, const Function<2> *> & function_map,
        const dealii::Quadrature<1> & quadrature
        )
{
    auto& fe = dof_handler.get_fe();

    FEFaceValues<2> fe_face_values (fe, quadrature,
                                      update_values         | update_quadrature_points | update_gradients |
                                      update_normal_vectors | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_face_q_points = quadrature.size();

    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_rhs = 0;

        for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        {
            if (!cell->face(face_number)->at_boundary())
                continue;
            //if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1))

            const Function<2>* boundary_value_func;
/*
            auto it = function_map.find(cell->face(face_number)->boundary_id());
            if (it == function_map.end())
                continue;

            boundary_value_func = it->second;*/

            fe_face_values.reinit(cell, face_number);

            for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    double v = fe_face_values.shape_grad(i, q_point) * fe_face_values.normal_vector(q_point) * // dphi/dn (q_i)
                            //boundary_value_func->value(fe_face_values.quadrature_point(q_point)) *
                            fe_face_values.quadrature_point(q_point)[1] *
                            fe_face_values.JxW(q_point);
                    cell_rhs(i) += (v);
                }
            }

        }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

void create_1_x_dphi_dx_dot_phi(
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
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    /*if (fabs(r) < 2e-3)
                        cell_matrix(i, j) = 0.0;
                    else*/
                        cell_matrix(i, j) += (
                            fe_values.shape_grad(i, q_index)[0] * // d phi_i(x_q) / dr
                            fe_values.shape_value(j, q_index) *   // phi_j(x_q)
                            1.0 / r *                             // 1/r
                            //r *                             // r
                            fe_values.JxW(q_index)                // dx
                        );
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
}

void create_phi_i_phi_j_dot_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor& tensor,
        const dealii::Quadrature<2> & quadrature)
{
    auto &fe = dof_handler.get_fe();

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    FullTensor cell_tensor(dofs_per_cell, dofs_per_cell, dofs_per_cell);

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
                            r *                                   // r
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
