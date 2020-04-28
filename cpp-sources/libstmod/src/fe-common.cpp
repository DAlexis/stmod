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
            std::cout << "Creating r-laplace matrix..." << std::endl;
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
            std::cout << "Creating r-mass matrix..." << std::endl;
            create_r_mass_matrix_axial(m_dof_handler,
                                           matrix);
            /*
            create_r_mass_matrix_axial(m_dof_handler,
                                       matrix,
                                       m_constraints,
                                       QGauss<2>(2 * m_fe.degree + 1));*/
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

void FEGlobalResources::add_subscriber(IFEGlobalResourcesUser* subscriber)
{
    m_subscribers.push_back(subscriber);
}

FEResources::FEResources(FEGlobalResources& global_resources) :
    m_global_resources(global_resources)
{
    global_resources.add_subscriber(this);

    m_r_laplace_matrix.set_initializer(
        [this](dealii::SparseMatrix<double>& r_laplace_matrix)
        {
            r_laplace_matrix.reinit(m_sparsity_pattern);
            std::cout << "Creating r-laplace matrix..." << std::endl;
            create_r_laplace_matrix_axial(m_global_resources.dof_handler(),
                                       r_laplace_matrix,
                                       m_constraints,
                                       QGauss<2>(2 * m_global_resources.degree() + 1));
        }
    );

    m_r_mass_matrix.set_initializer(
        [this](dealii::SparseMatrix<double>& r_mass_matrix)
        {
            r_mass_matrix.reinit(m_sparsity_pattern);
            std::cout << "Creating r-mass matrix..." << std::endl;
            create_r_mass_matrix_axial(m_global_resources.dof_handler(),
                                       r_mass_matrix,
                                       m_constraints,
                                       QGauss<2>(2 * m_global_resources.degree() + 1));
        }
    );

    m_inverse_r_mass_matrix.set_initializer(
        [this](dealii::SparseDirectUMFPACK& inverse_r_mass_matrix)
        {
            std::cout << "Creating inverse_r_mass_matrix..." << std::endl;
            inverse_r_mass_matrix.initialize(m_r_mass_matrix.get());
        }
    );

    m_inverse_r_laplace_matrix.set_initializer(
        [this](dealii::SparseDirectUMFPACK& inverse_r_laplace_matrix)
        {
            std::cout << "Creating inverse_r_laplace_matrix..." << std::endl;
            inverse_r_laplace_matrix.initialize(m_r_laplace_matrix.get());
        }
    );

    m_phi_i_phi_j_dot_r_phi_k.set_initializer(
        [this](SparseTensor3& phi_i_phi_j_dot_r_phi_k)
        {
            phi_i_phi_j_dot_r_phi_k.clear();

            std::cout << "Creating phi_i_phi_j_dot_r_phi_k tensor..." << std::endl;
            create_phi_i_phi_j_dot_r_phi_k(
                    m_global_resources.dof_handler(),
                    phi_i_phi_j_dot_r_phi_k,
                    QGauss<2>(3 * m_global_resources.degree() + 1));
        }
    );

    m_grad_phi_i_grad_phi_j_dot_r_phi_k.set_initializer(
        [this](SparseTensor3& grad_phi_i_grad_phi_j_dot_r_phi_k)
        {
            grad_phi_i_grad_phi_j_dot_r_phi_k.clear();
            std::cout << "Creating grad_phi_i_grad_phi_j_dot_r_phi_k tensor..." << std::endl;
            create_grad_phi_i_grad_phi_j_dot_r_phi_k(
                    m_global_resources.dof_handler(),
                    grad_phi_i_grad_phi_j_dot_r_phi_k,
                    QGauss<2>(3 * m_global_resources.degree() + 1));
        }
    );
}

dealii::AffineConstraints<double>& FEResources::constraints()
{
    return m_constraints;
}

void FEResources::on_triangulation_updated()
{
    m_cleaner.clear();
    m_constraints.clear();
        DoFTools::make_hanging_node_constraints(m_global_resources.dof_handler(), m_constraints);
        if (m_boundary_conditions_maker_func)
            m_boundary_conditions_maker_func(m_constraints);
    m_constraints.close();

    DynamicSparsityPattern dsp(m_global_resources.n_dofs());
    DoFTools::make_sparsity_pattern(m_global_resources.dof_handler(),
                                  dsp,
                                  m_constraints,
                                  /*keep_constrained_dofs = */ true);
    m_sparsity_pattern.copy_from(dsp);

    std::cout << "Single-fraction problem has " << m_global_resources.n_dofs() << " dimensions of freedom" << std::endl;
}

void FEResources::set_boundary_cond_gen(BoundaryConditionsGeneratorFunc gen)
{
    m_boundary_conditions_maker_func = gen;
}

const FEGlobalResources& FEResources::global_resources() const
{
    return m_global_resources;
}

const dealii::SparseMatrix<double>& FEResources::r_laplace_matrix_axial() const
{
    return m_r_laplace_matrix;
}

const dealii::SparseMatrix<double>& FEResources::r_mass_matrix() const
{
    return m_r_mass_matrix;
}

const dealii::SparseDirectUMFPACK& FEResources::inverse_r_mass_matrix() const
{
    return m_inverse_r_mass_matrix;
}

const dealii::SparseDirectUMFPACK& FEResources::inverse_r_laplace_matrix() const
{
    return m_inverse_r_laplace_matrix;
}

const SparseTensor3& FEResources::phi_i_phi_j_dot_r_phi_k() const
{
    return m_phi_i_phi_j_dot_r_phi_k;
}

const SparseTensor3& FEResources::grad_phi_i_grad_phi_j_dot_r_phi_k() const
{
    return m_grad_phi_i_grad_phi_j_dot_r_phi_k;
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
