#include "stmod/fe-common.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <iostream>

using namespace dealii;

FEGlobalResources::FEGlobalResources(dealii::Triangulation<2>& triangulation, unsigned int degree) :
    m_triangulation(triangulation), m_fe(degree), m_dof_handler(m_triangulation)
{
    m_laplace_matrix.set_initializer(
        [this](dealii::SparseMatrix<double>& matrix)
        {
            matrix.reinit(m_sparsity_pattern);
            //std::cout << "Creating r-laplace matrix..." << std::endl;
            create_r_laplace_matrix_axial(m_dof_handler,
                                           matrix);
            /*

            create_r_laplace_matrix_axial(m_dof_handler,
                                       matrix,
                                       m_constraints,
                                       QGauss<2>(2 * m_fe.degree + 1));*/
        }
    );

    m_mass_matrix.set_initializer(
        [this](dealii::SparseMatrix<double>& matrix)
        {
            matrix.reinit(m_sparsity_pattern);
            //std::cout << "Creating r-mass matrix..." << std::endl;
            create_r_mass_matrix_axial(m_dof_handler,
                                           matrix);
            /*
            create_r_mass_matrix_axial(m_dof_handler,
                                       matrix,
                                       m_constraints,
                                       QGauss<2>(2 * m_fe.degree + 1));*/
        }
    );

    m_grad_phi_i_grad_phi_j_dot_r_phi_k.set_initializer(
        [this](SparseTensor3& tensor)
        {
            tensor.clear();
            //std::cout << "Creating grad_phi_i_grad_phi_j_dot_r_phi_k tensor..." << std::endl;
            create_grad_phi_i_grad_phi_j_dot_r_phi_k(
                    m_dof_handler,
                    tensor);
        }
    );
}

dealii::Triangulation<2>& FEGlobalResources::triangulation()
{
    return m_triangulation;
}

void FEGlobalResources::on_triangulation_updated()
{
    m_cleaner.clear();
    m_dof_handler.distribute_dofs(m_fe);
    m_constraints.clear();
        // No boundary values inside constraints
        DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    m_constraints.close();

    DynamicSparsityPattern dsp(n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler,
                                  dsp,
                                  m_constraints,
                                  /*keep_constrained_dofs = */ true);

    m_sparsity_pattern.copy_from(dsp);

    for (auto subscriber : m_subscribers)
        subscriber->on_triangulation_updated();
}

unsigned int FEGlobalResources::degree() const
{
    return m_fe.degree;
}


const dealii::FE_Q<2>& FEGlobalResources::fe() const
{
    return m_fe;
}

const dealii::DoFHandler<2>& FEGlobalResources::dof_handler() const
{
    return m_dof_handler;
}

dealii::types::global_dof_index FEGlobalResources::n_dofs() const
{
    return m_dof_handler.n_dofs();
}

const dealii::SparsityPattern& FEGlobalResources::sparsity_pattern() const
{
    return m_sparsity_pattern;
}


const dealii::AffineConstraints<double>& FEGlobalResources::constraints() const
{
    return m_constraints;
}

const dealii::SparseMatrix<double>& FEGlobalResources::mass_matrix() const
{
    return m_mass_matrix;
}

const dealii::SparseMatrix<double>& FEGlobalResources::laplace_matrix() const
{
    return m_laplace_matrix;
}

const SparseTensor3& FEGlobalResources::grad_phi_i_grad_phi_j_dot_r_phi_k() const
{
    return m_grad_phi_i_grad_phi_j_dot_r_phi_k;
}

void FEGlobalResources::add_subscriber(IFEGlobalResourcesUser* subscriber)
{
    m_subscribers.push_back(subscriber);
}
