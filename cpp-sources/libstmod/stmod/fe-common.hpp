#ifndef FE_COMMON_HPP_INCLUDED
#define FE_COMMON_HPP_INCLUDED

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
    const dealii::SparseMatrix<double>& laplace_addition_matrix() const;
    const dealii::SparseMatrix<double>& r_laplace_matrix_axial() const;
    const dealii::SparseMatrix<double>& mass_matrix() const;
    const dealii::SparsityPattern& sparsity_pattern() const;
    const dealii::AffineConstraints<double>& constraints() const;

private:
    dealii::Triangulation<2>& m_triangulation;
    dealii::AffineConstraints<double> m_constraints;

    dealii::FE_Q<2>        m_fe;
    dealii::DoFHandler<2>  m_dof_handler;
    dealii::SparsityPattern  m_sparsity_pattern;
    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_solution;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<float>  m_estimated_error_per_cell;

    dealii::SparseMatrix<double> m_mass_matrix;
    dealii::SparseMatrix<double> m_laplace_matrix;
    dealii::SparseMatrix<double> m_laplace_addition_matrix;
    dealii::SparseMatrix<double> m_r_laplace_matrix_axial;
};

#endif // FE_COMMON_HPP_INCLUDED
