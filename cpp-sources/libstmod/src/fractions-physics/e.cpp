#include "stmod/fractions-physics/e.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>

using namespace dealii;


Electrons::Electrons(const FEGlobalResources& fe_global_res) :
    m_fe_global_res(fe_global_res)
{
}

const std::string& Electrons::output_name(size_t index) const
{
    return m_names[index];
}

const dealii::Vector<double>& Electrons::output_value(size_t index) const
{
    switch (index)
    {
    case 0:
        return m_concentration;
    case 1:
        return m_derivative;
    default:
        throw std::range_error("Electrons::value(): invalid quantity index");
    };
}

size_t Electrons::output_values_count() const
{
    return 2;
}

dealii::Vector<double>& Electrons::values_vector()
{
    return m_concentration;
}

const dealii::Vector<double>& Electrons::derivatives_vector() const
{
    return m_derivative;
}

void Electrons::compute(double t)
{
    create_rhs();
    solve_lin_eq();
    add_single_point_derivative();
}

const dealii::Vector<double>& Electrons::derivative() const
{
    return m_derivative;
}

dealii::Vector<double>& Electrons::value_w()
{
    return m_concentration;
}

void Electrons::add_single_source(double reaction_const, const dealii::Vector<double>& source)
{
    m_single_sources.push_back(&source);
    m_single_reaction_consts.push_back(reaction_const);
}

void Electrons::add_pair_source(double reaction_const, const dealii::Vector<double>& source1, const dealii::Vector<double>& source2)
{
    m_pair_sources.push_back(PairSourceTuple(&source1, &source2));
    m_pair_reaction_consts.push_back(reaction_const);
}

void Electrons::set_potential_and_total_charge(const dealii::Vector<double>& potential, const dealii::Vector<double>& total_charge)
{
    m_potential = &potential;
    m_total_charge = &total_charge;
}

const dealii::Vector<double>& Electrons::get_implicit_dn(double dt, double theta)
{
    create_implicit_method_matrixes(dt, theta);
    m_tmp = 0;
    m_implicit_rhs_matrix.vmult(m_tmp, m_concentration);
    m_implicit_delta = 0;
    m_implicit_system_reversed.vmult(m_implicit_delta, m_tmp);
    m_constraints.distribute(m_implicit_delta);
    return m_implicit_delta;
}

void Electrons::init_mesh_dependent()
{
    m_constraints.clear();
        DoFTools::make_hanging_node_constraints(m_fe_global_res.dof_handler(), m_constraints);
    m_constraints.close();

    DynamicSparsityPattern dsp(m_fe_global_res.n_dofs());
    DoFTools::make_sparsity_pattern(m_fe_global_res.dof_handler(),
                                  dsp,
                                  m_constraints,
                                  /*keep_constrained_dofs = */ false);
    m_sparsity_pattern.copy_from(dsp);

    m_system_rhs.reinit(m_fe_global_res.n_dofs());

    m_concentration.reinit(m_fe_global_res.n_dofs());
    m_derivative.reinit(m_fe_global_res.n_dofs());
    m_derivative_without_single_point.reinit(m_fe_global_res.n_dofs());

    m_implicit_delta.reinit(m_fe_global_res.n_dofs());
    m_tmp.reinit(m_fe_global_res.n_dofs());

    m_E_grad_psi_psi_matrix.reinit(m_sparsity_pattern);
    m_mass_matrix_axial.reinit(m_sparsity_pattern);
    m_laplace_matrix_axial.reinit(m_sparsity_pattern);


    m_implicit_rhs_matrix.reinit(m_sparsity_pattern);
    m_implicit_system_matrix.reinit(m_sparsity_pattern);

    create_E_grad_psi_psi_matrix_axial(
            10/0.002, 10/0.002,
            m_fe_global_res.dof_handler(),
            m_E_grad_psi_psi_matrix,
            m_constraints,
            dealii::QGauss<2>(2 * m_fe_global_res.fe().degree - 1));

    create_r_mass_matrix_axial(
                m_fe_global_res.dof_handler(),
                m_mass_matrix_axial,
                m_constraints,
                dealii::QGauss<2>(2 * m_fe_global_res.fe().degree - 1));

    create_r_laplace_matrix_axial(
                m_fe_global_res.dof_handler(),
                m_laplace_matrix_axial,
                m_constraints,
                dealii::QGauss<2>(2 * m_fe_global_res.fe().degree - 1));


}

const dealii::Vector<double>& Electrons::error_estimation_vector() const
{
    return m_concentration;
}

void Electrons::create_implicit_method_matrixes(double dt, double theta)
{
    m_implicit_system_matrix.copy_from(m_mass_matrix_axial);
    m_implicit_system_matrix.add(-dt*theta*parameters.mu_e, m_E_grad_psi_psi_matrix);
    m_implicit_system_matrix.add(-dt*theta*parameters.D_e, m_laplace_matrix_axial);

    m_implicit_rhs_matrix = 0;
    m_implicit_rhs_matrix.add(dt*parameters.mu_e, m_E_grad_psi_psi_matrix);
    m_implicit_rhs_matrix.add(dt*parameters.D_e, m_laplace_matrix_axial);

    m_implicit_system_reversed.initialize(m_implicit_system_matrix);
    //create_E_grad_psi_psi_matrix_axial()
}

void Electrons::create_rhs()
{    
    m_system_rhs = 0;
    /*
    if (m_potential)
        sum_with_tensor(m_system_rhs, m_concentration, *m_potential, m_fe_res.grad_phi_i_grad_phi_j_dot_r_phi_k());

*/

    m_E_grad_psi_psi_matrix.vmult(m_system_rhs, m_concentration);
    m_system_rhs *= -parameters.mu_e;
    m_constraints.distribute(m_system_rhs);

    /*
    m_tmp = 0;
    m_fe_res.r_laplace_matrix_axial().vmult(m_tmp, m_concentration);
    //m_fe_res.laplace_matrix().vmult(m_tmp, m_concentration);

    m_system_rhs.add(parameters.D_e, m_tmp);*/
}

void Electrons::solve_lin_eq()
{
    //m_fe_res.inverse_r_mass_matrix().vmult(m_derivative_without_single_point, m_system_rhs);
    /*m_fe_res.lin_eq_solver().solve(
                m_system_matrix, m_derivative_without_single_point, m_system_rhs,
                1e-8, "Electrons");*/
}

void Electrons::add_single_point_derivative()
{
    m_derivative = m_derivative_without_single_point;

    // Adding charges effect at this point: mu_e * n_e * Q
    /*if (m_potential)
    {
        m_tmp = m_concentration;
        m_tmp.scale(*m_total_charge);
        m_derivative.add(parameters.mu_e, m_tmp);
    }*/

    for (size_t i = 0; i < m_single_sources.size(); i++)
    {
        m_derivative.add(m_single_reaction_consts[i], (*m_single_sources[i]));
    }

    for (size_t i = 0; i < m_pair_sources.size(); i++)
    {
        m_tmp = *std::get<0>(m_pair_sources[i]) * *std::get<1>(m_pair_sources[i]);
        m_derivative.add(m_pair_reaction_consts[i], m_tmp);
    }
}
