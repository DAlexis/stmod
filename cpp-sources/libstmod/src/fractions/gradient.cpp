#include "stmod/fractions/gradient.hpp"

Gradient::Gradient(
        const FEGlobalResources& fe_res,
        const dealii::Vector<double>& field,
        const std::string& name,
        size_t component,
        double multiplier) :
    m_fe_global_res(fe_res),
    m_field(field),
    m_component(component),
    m_multiplier(multiplier),
    ScalarVariable(name)
{
}

// IMeshBased
void Gradient::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);

    m_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());
}

// IPreStepComputer
void Gradient::compute(double)
{
    m_value = 0;
    m_rhs = 0;

    m_fe_global_res.r_grad_phi_i_comp_phi_j(m_component).vmult(m_rhs, m_field);
    m_fe_global_res.inverse_mass_matrix().vmult(m_value, m_rhs);
    m_fe_global_res.constraints().distribute(m_value);
    m_value *= m_multiplier;
}
