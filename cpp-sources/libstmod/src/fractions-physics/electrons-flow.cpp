#include "stmod/fractions-physics/electrons-flow.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/numerics/matrix_tools.h>

ElectronsFlow::ElectronsFlow(
        const FEGlobalResources& fe_res,
        const dealii::Vector<double>& concentration,
        const dealii::Vector<double>& E_field_component,
        size_t component,
        ElectronsFlowParameters flow_parameters) :
    m_fe_global_res(fe_res),
    m_flow_parameters(flow_parameters),
    m_n_e(concentration),
    m_E_component(E_field_component),
    m_component(component),
    ScalarVariable(std::string("electrons_flow_") + std::to_string(component))
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
    // First computing diffusion flow
    m_value = 0;

    // Creating rhs matrix
    m_rhs_matrix = 0;

    // Diffusion part
    m_rhs = 0;
    m_fe_global_res.r_grad_phi_i_comp_phi_j(m_component).vmult(m_rhs, m_n_e);
    m_fe_global_res.inverse_mass_matrix().vmult(m_value, m_rhs);
    m_value /= -m_flow_parameters.D;

    // Drifting flow

    for (dealii::Vector<double>::size_type i = 0; i < m_value.size(); i++)
    {
        m_value[i] += -m_flow_parameters.mu * m_n_e[i] * m_E_component[i];
    }

    m_fe_global_res.constraints().distribute(m_value);
}
