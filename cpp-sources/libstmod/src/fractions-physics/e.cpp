#include "stmod/fractions-physics/e.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/matgen.hpp"

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

const std::string& Electrons::name(size_t index) const
{
    return m_names[index];
}

const dealii::Vector<double>& Electrons::value(size_t index) const
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

size_t Electrons::values_count() const
{
    return 2;
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

void Electrons::set_potential(const dealii::Vector<double>& potential)
{
    m_potential = &potential;
}

void Electrons::init()
{
    m_system_rhs.reinit(m_fe_res.dof_handler().n_dofs());

    m_concentration.reinit(m_fe_res.dof_handler().n_dofs());
    m_derivative.reinit(m_fe_res.dof_handler().n_dofs());
    m_derivative_without_single_point.reinit(m_fe_res.dof_handler().n_dofs());
    m_tmp.reinit(m_fe_res.dof_handler().n_dofs());

    m_system_matrix.reinit(m_fe_res.sparsity_pattern());

    m_system_matrix.copy_from(m_fe_res.r_mass_matrix());
}

void Electrons::solve()
{
    assemble_system();
    solve_lin_eq();
    add_single_point_derivative();
}


void Electrons::assemble_system()
{    
    m_system_rhs = 0;
    if (m_potential)
        sum_with_tensor(m_system_rhs, m_concentration, *m_potential, m_fe_res.grad_phi_i_grad_phi_j_dot_r_phi_k());

    m_system_rhs *= parameters.mu_e;

    m_tmp = 0;
    m_fe_res.r_laplace_matrix_axial().vmult(m_tmp, m_concentration);
    //m_fe_res.laplace_matrix().vmult(m_tmp, m_concentration);

    m_system_rhs.add(parameters.D_e, m_tmp);
}

void Electrons::solve_lin_eq()
{
    m_fe_res.lin_eq_solver().solve(
                m_system_matrix, m_derivative_without_single_point, m_system_rhs,
                5e-3, "Electrons");
}

void Electrons::add_single_point_derivative()
{
    m_derivative = m_derivative_without_single_point;
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
