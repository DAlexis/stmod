#include "stmod/fe-common.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>

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
    m_laplace_addition_matrix.reinit(m_sparsity_pattern);
    m_r_laplace_matrix_axial.reinit(m_sparsity_pattern);

    create_mass_matrix_axial(m_dof_handler,
                            m_mass_matrix,
                            m_constraints,
                            QGauss<2>(m_fe.degree + 1));

    create_laplace_matrix_axial(m_dof_handler,
                               m_laplace_matrix,
                               m_constraints,
                               QGauss<2>(m_fe.degree + 1));

    create_1_x_dphi_dx_dot_phi(m_dof_handler,
                               m_laplace_addition_matrix,
                               m_constraints,
                               QGauss<2>(2*m_fe.degree));

    create_r_laplace_matrix_axial(m_dof_handler,
                               m_r_laplace_matrix_axial,
                               m_constraints,
                               QGauss<2>(m_fe.degree + 1));



    /*MatrixCreator::create_laplace_matrix(m_dof_handler,
                                   QGauss<2>(m_fe.degree + 1),
                                   m_laplace_matrix);*/
}

const dealii::DoFHandler<2>& FEResources::dof_handler() const
{
    return m_dof_handler;
}

const dealii::SparseMatrix<double>& FEResources::laplace_matrix() const
{
    return m_laplace_matrix;
}

const dealii::SparseMatrix<double>& FEResources::laplace_addition_matrix() const
{
    return m_laplace_addition_matrix;
}

const dealii::SparseMatrix<double>& FEResources::r_laplace_matrix_axial() const
{
    return m_r_laplace_matrix_axial;
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
