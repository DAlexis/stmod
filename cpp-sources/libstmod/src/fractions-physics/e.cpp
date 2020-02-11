#include "stmod/fractions-physics/e.hpp"
#include "stmod/fe-sampler.hpp"

ElectronsRHS::ElectronsRHS(
        FractionsStorage& storage,
        size_t fraction_index,
        const dealii::Vector<double>& potential_solution,
        const dealii::DoFHandler<2>& dof_handler
    ) :
    FractionBase(storage, fraction_index),
    m_potential(potential_solution),
    m_dof_handler(dof_handler)
{
}

void ElectronsRHS::calculate_rhs(double time)
{
    FESampler field_sampler(m_dof_handler), electrons_sampler(m_dof_handler);

    const dealii::Vector<double>& ne_current = m_storage.current(m_fraction_index);
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
        ne_rhs[i] = 1e6;//constants.D_e * n_laplacians[i];
    }
}


ElectronIterator::ElectronIterator(
        FractionsStorage& storage,
        size_t fraction_index,
        const dealii::Vector<double>& potential_solution,
        const dealii::DoFHandler<2>& dof_handler
    ) :
    FractionBase(storage, fraction_index),
    //m_potential(potential_solution),
    m_dof_handler(dof_handler)
{
}


Electrons::Electrons(const FEResources& fe_res) :
    m_fe_res(fe_res)
{
}

const std::string& Electrons::name() const
{
    return m_name;
}

const dealii::Vector<double>& Electrons::value() const
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

void Electrons::init()
{
    /// @todo Here
}

void Electrons::assemble_system()
{
    /// @todo Here
}

void Electrons::solve_lin_eq()
{
    /// @todo Here
}
