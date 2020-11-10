#include "stmod/fractions-physics/electrons-flow.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/numerics/matrix_tools.h>

ElectronsFlow::ElectronsFlow(const FEGlobalResources& fe_res, size_t component, ElectronsFlowParameters flow_parameters) :
    m_fe_global_res(fe_res), m_flow_parameters(flow_parameters), m_component(component), ScalarVariable(std::string("electrons_flow_") + std::to_string(component))
{
}

void ElectronsFlow::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);
    m_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());

    m_tmp_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_system_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());

    m_boundary_values.clear();
}

void ElectronsFlow::compute(double)
{
    m_value = 0;

    // Creating rhs matrix
    m_rhs_matrix = 0;

    // Drifting part
    m_tmp_matrix = 0;
    create_E_psi_psi_matrix(
            *m_E_component,
            m_fe_global_res.dof_handler(),
            m_tmp_matrix);

    m_rhs_matrix.add(-m_flow_parameters.mu, m_tmp_matrix);

    // Diffusion part
    m_tmp_matrix = 0;
    create_r_grad_phi_i_comp_phi_j_axial(
            m_fe_global_res.dof_handler(),
            m_tmp_matrix,
            m_component);

    m_rhs_matrix.add(-m_flow_parameters.D, m_tmp_matrix);

    //m_rhs_matrix.add(m_flow_parameters.D, m_fe_global_res.laplace_matrix()); // TODO: not a Laplace matrix! fix this!!

    // Creating rhs
    m_rhs_matrix.vmult(m_rhs, *m_n_e);

    // Preparing to solve system
    m_system_matrix.copy_from(m_fe_global_res.mass_matrix());
    m_fe_global_res.constraints().condense(m_system_matrix, m_rhs);

    // Applying boundary conditions
    dealii::MatrixTools::apply_boundary_values(m_boundary_values,
                                     m_system_matrix,
                                     m_value,
                                     m_rhs);

    // Solving system
    m_system_matrix_inverse.initialize(m_system_matrix);
    m_system_matrix_inverse.vmult(m_value, m_rhs);

    m_fe_global_res.constraints().distribute(m_value);
}

void ElectronsFlow::set_electric_field(const dealii::Vector<double>& E_component)
{
    m_E_component = &E_component;
}

void ElectronsFlow::set_electrons_density(const dealii::Vector<double>& n_e)
{
    m_n_e = &n_e;
}
