#ifndef MATGEN_HPP_INCLUDED
#define MATGEN_HPP_INCLUDED

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/quadrature_lib.h>

#include <unordered_map>

struct FullTensor
{
public:
    FullTensor(unsigned int size_i, unsigned int size_j, unsigned int size_k);

    double& operator()(unsigned int i, unsigned int j, unsigned int k);
    void operator=(double x);

private:
    unsigned int m_size_i, m_size_j, m_size_k;

    std::vector<double> m_content;
};


struct SparseTensor
{
public:
    double operator()(unsigned int i, unsigned int j, unsigned int k);
    void set(unsigned int i, unsigned int j, unsigned int k, double value);

private:
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, std::unordered_map<unsigned int, double>>> m_unordered_map;
};

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

void create_mass_matrix_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 3));

void add_dirichlet_rhs_axial(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::Vector<double>& system_rhs,
        const std::map<dealii::types::boundary_id, const dealii::Function<2>*> & function_map,
        const dealii::Quadrature<1> & quadrature = dealii::QGauss<1>(/*degree = */ 3)
        );






void create_1_x_dphi_dx_dot_phi(
        const dealii::DoFHandler<2, 2>& dof_handler,
        dealii::SparseMatrix<double> &sparse_matrix,
        const dealii::AffineConstraints<double> & constraints = dealii::AffineConstraints<double>(),
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 10));


void create_phi_i_phi_j_dot_phi_k(
        const dealii::DoFHandler<2, 2>& dof_handler,
        SparseTensor& tensor,
        const dealii::Quadrature<2> & quadrature = dealii::QGauss<2>(/*degree = */ 4));

#endif // MATGEN_HPP_INCLUDED
