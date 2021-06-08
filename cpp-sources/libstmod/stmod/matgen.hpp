#ifndef MATGEN_HPP_INCLUDED
#define MATGEN_HPP_INCLUDED

#include "tensors.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/quadrature_lib.h>

constexpr static double default_r_epsilon = 1e-6;
//constexpr static double default_r_epsilon = 0;

void create_E_psi_psi_matrix(
        const dealii::Vector<double>& E,
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_E_grad_psi_psi_matrix_axial(
        double Ex, double Ey,
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_E_grad_psi_psi_matrix_axial(
        const dealii::Vector<double>& potential,
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_E_psi_grad_psi_matrix_axial(
        const dealii::Vector<double>& Ex,
        const dealii::Vector<double>& Ey,
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 5),
        double r_epsilon = default_r_epsilon);


// Version without rhs
void create_r_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

/**
 * @brief Function version with RHS. This means function creates RHS vector with ONLY boundary
 * conditions and zeros in anoter places
 * @param dof_handler
 * @param sparse_matrix
 * @param rhs
 * @param constraints
 * @param quadrature
 * @param r_epsilon
 */
void create_r_laplace_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        dealii::Vector<double>& rhs,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_r_mass_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_r_grad_phi_i_comp_phi_j_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        unsigned int component,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3),
        double r_epsilon = default_r_epsilon);

void create_phi_i_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 4),
        double r_epsilon = default_r_epsilon);

void create_grad_phi_i_grad_phi_j_dot_r_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor3& tensor,
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 4),
        double r_epsilon = default_r_epsilon);

void sum_with_tensor(dealii::Vector<double>& out_vector,
                     const dealii::Vector<double>& in_first,
                     const dealii::Vector<double>& in_second,
                     const SparseTensor3& tensor);

#endif // MATGEN_HPP_INCLUDED
