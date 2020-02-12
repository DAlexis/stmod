#ifndef FE_COMMON_HPP_INCLUDED
#define FE_COMMON_HPP_INCLUDED

#include "lin-eq-solver.hpp"
#include "tensors.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

class FEResources
{
public:
    FEResources(dealii::Triangulation<2>& triangulation, unsigned int degree);

    void init();

    const dealii::DoFHandler<2>& dof_handler() const;
    const dealii::SparseMatrix<double>& laplace_matrix() const;

    /**
     * @brief Matrix for (d/dr(psi_i), psi_j) + (grad psi_i, grad(r * psi_j)),
     * so this is matrix for [r * laplacian_in_axial_coordinates]
     */
    const dealii::SparseMatrix<double>& r_laplace_matrix_axial() const;
    /**
     * @brief Matrix for (psi_i, r*psi_j)
     *
     */
    const dealii::SparseMatrix<double>& r_mass_matrix() const;

    const SparseTensor3& phi_i_phi_j_dot_r_phi_k() const;
    const SparseTensor3& grad_phi_i_grad_phi_j_dot_r_phi_k() const;

    const dealii::SparseMatrix<double>& mass_matrix() const;
    const dealii::SparsityPattern& sparsity_pattern() const;
    const dealii::AffineConstraints<double>& constraints() const;
    const LinEqSolver& lin_eq_solver() const;

private:
    dealii::Triangulation<2>& m_triangulation;
    dealii::AffineConstraints<double> m_constraints;
    LinEqSolver m_lin_eq_solver{m_constraints, 1e-2};

    dealii::FE_Q<2>        m_fe;
    dealii::DoFHandler<2>  m_dof_handler;
    dealii::SparsityPattern  m_sparsity_pattern;
    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_solution;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<float>  m_estimated_error_per_cell;

    dealii::SparseMatrix<double> m_mass_matrix;
    dealii::SparseMatrix<double> m_laplace_matrix;
    dealii::SparseMatrix<double> m_r_laplace_matrix_axial;
    dealii::SparseMatrix<double> m_r_mass_matrix;

    SparseTensor3 m_phi_i_phi_j_dot_r_phi_k;
    SparseTensor3 m_grad_phi_i_grad_phi_j_dot_r_phi_k;
};

#endif // FE_COMMON_HPP_INCLUDED
