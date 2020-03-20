#include "stmod/fe-common.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <iostream>

using namespace dealii;

FEResources::FEResources(dealii::Triangulation<2>& triangulation, unsigned int degree) :
    m_triangulation(triangulation), m_fe(degree), m_dof_handler(m_triangulation)
{
}

void FEResources::init()
{
    m_dof_handler.distribute_dofs(m_fe);

    m_constraints.clear();
        DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    m_constraints.close();

    DynamicSparsityPattern dsp(m_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler,
                                  dsp,
                                  m_constraints,
                                  /*keep_constrained_dofs = */ true);
    m_sparsity_pattern.copy_from(dsp);

    m_system_matrix.reinit(m_sparsity_pattern);

    m_mass_matrix.reinit(m_sparsity_pattern);
    m_laplace_matrix.reinit(m_sparsity_pattern);
    m_r_laplace_matrix_axial.reinit(m_sparsity_pattern);
    m_r_mass_matrix.reinit(m_sparsity_pattern);

    m_phi_i_phi_j_dot_r_phi_k.clear();
    m_grad_phi_i_grad_phi_j_dot_r_phi_k.clear();



    std::cout << "Single-fraction problem has " << m_dof_handler.n_dofs() << " dimensions of freedom" << std::endl;
    std::cout << "Creating mass matrix..." << std::endl;
    MatrixTools::create_mass_matrix(m_dof_handler,
                            QGauss<2>(2*m_fe.degree),
                            m_mass_matrix,
                            static_cast<const Function<2> *const>(nullptr),
                            m_constraints);

    std::cout << "Creating laplace matrix..." << std::endl;
    MatrixTools::create_laplace_matrix(m_dof_handler,
                            QGauss<2>(2 * m_fe.degree),
                            m_laplace_matrix,
                            static_cast<const Function<2> *const>(nullptr),
                            m_constraints);

    std::cout << "Creating r-laplace matrix..." << std::endl;
    create_r_laplace_matrix_axial(m_dof_handler,
                               m_r_laplace_matrix_axial,
                               m_constraints,
                               QGauss<2>(2 * m_fe.degree + 1));

    std::cout << "Creating r-mass matrix..." << std::endl;
    create_r_mass_matrix_axial(m_dof_handler,
                               m_r_mass_matrix,
                               m_constraints,
                               QGauss<2>(2 * m_fe.degree + 1));

    std::cout << "Creating phi_i_phi_j_dot_r_phi_k tensor..." << std::endl;
    create_phi_i_phi_j_dot_r_phi_k(
            m_dof_handler,
            m_phi_i_phi_j_dot_r_phi_k,
            QGauss<2>(3 * m_fe.degree + 1));

    std::cout << "Creating grad_phi_i_grad_phi_j_dot_r_phi_k tensor..." << std::endl;
    create_grad_phi_i_grad_phi_j_dot_r_phi_k(
            m_dof_handler,
            m_grad_phi_i_grad_phi_j_dot_r_phi_k,
            QGauss<2>(3 * m_fe.degree + 1));

    std::cout << "Creating m_inverse_r_mass_matrix..." << std::endl;
    m_inverse_r_mass_matrix.initialize(m_r_mass_matrix);
    std::cout << "All matrixes and tensors created" << std::endl;
}

const dealii::DoFHandler<2>& FEResources::dof_handler() const
{
    return m_dof_handler;
}

const dealii::SparseMatrix<double>& FEResources::laplace_matrix() const
{
    return m_laplace_matrix;
}

const dealii::SparseMatrix<double>& FEResources::r_laplace_matrix_axial() const
{
    return m_r_laplace_matrix_axial;
}

const dealii::SparseMatrix<double>& FEResources::r_mass_matrix() const
{
    return m_r_mass_matrix;
}

const dealii::SparseDirectUMFPACK& FEResources::inverse_r_mass_matrix() const
{
    return m_inverse_r_mass_matrix;
}

const SparseTensor3& FEResources::phi_i_phi_j_dot_r_phi_k() const
{
    return m_phi_i_phi_j_dot_r_phi_k;
}

const SparseTensor3& FEResources::grad_phi_i_grad_phi_j_dot_r_phi_k() const
{
    return m_grad_phi_i_grad_phi_j_dot_r_phi_k;
}

const dealii::SparseMatrix<double>& FEResources::mass_matrix() const
{
    return m_mass_matrix;
}

const dealii::SparsityPattern& FEResources::sparsity_pattern() const
{
    return m_sparsity_pattern;
}

const dealii::AffineConstraints<double>& FEResources::constraints() const
{
    return m_constraints;
}

const LinEqSolver& FEResources::lin_eq_solver() const
{
    return m_lin_eq_solver;
}
