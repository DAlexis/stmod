#ifndef MATGEN_HPP_INCLUDED
#define MATGEN_HPP_INCLUDED

#include "tensors.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/quadrature_lib.h>

void create_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3));

void create_r_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3));

void create_r_mass_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3));

void create_mass_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3));

void create_phi_i_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 4));

void create_grad_phi_i_grad_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 4));

void sum_with_tensor(dealii::Vector<double>& out_vector,
                     const dealii::Vector<double>& in_first,
                     const dealii::Vector<double>& in_second,
                     const SparseTensor3& tensor);

#endif // MATGEN_HPP_INCLUDED
