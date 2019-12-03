#include "stmod/fractions-physics/e.hpp"
#include "stmod/fe-sampler.hpp"

ElectronsRHS::ElectronsRHS(
        FractionsStorage& storage,
        size_t fraction_index,
        const dealii::Vector<double>& potential_solution,
        const dealii::DoFHandler<2>& dof_handler
    ) :
    FractionRHSBase(storage, fraction_index),
    m_potential(potential_solution),
    m_dof_handler(dof_handler)
{
}

void ElectronsRHS::calculate_rhs(double time)
{
    FESampler field_sampler(m_dof_handler), electrons_sampler(m_dof_handler);

    auto ne_current = m_storage.current(m_fraction_index);
    dealii::Vector<double>& ne_rhs = m_storage.rhs(m_fraction_index);

    // Sampling electric field gradient
    field_sampler.sample(m_potential);
    // Sampling n_e gradient and laplacian
    electrons_sampler.sample(ne_current);

    const std::vector<dealii::Point<2>>& e_field = field_sampler.gradients();
    const std::vector<dealii::Point<2>>& n_grad = electrons_sampler.gradients();
    const std::vector<double>& n_laplacians =  electrons_sampler.laplacians();

    for (unsigned int i = 0; i < ne_current.size(); i++)
    {
        ne_rhs[i] = constants.D_e * n_laplacians[i];
    }
}
