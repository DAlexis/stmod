#include "stmod/fractions/fraction.hpp"


Fraction::Fraction(const std::string& name) :
    m_name(name)
{
}

void Fraction::add_single_source(double reaction_const, const dealii::Vector<double>& source)
{
    m_single_sources.push_back(&source);
    m_single_reaction_consts.push_back(reaction_const);
}

void Fraction::add_pair_source(double reaction_const, const dealii::Vector<double>& source1, const dealii::Vector<double>& source2)
{
    m_pair_sources.push_back(PairSourceTuple(&source1, &source2));
    m_pair_reaction_consts.push_back(reaction_const);
}

dealii::Vector<double>& Fraction::values_w()
{
    return m_concentration;
}

const dealii::Vector<double>& Fraction::derivatives() const
{
    return m_derivative;
}

const std::string& Fraction::output_name(size_t index) const
{
    return m_name;
}

const dealii::Vector<double>& Fraction::output_value(size_t index) const
{
    return m_concentration;
}

size_t Fraction::output_values_count() const
{
    return 1;
}

void Fraction::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    m_derivative.reinit(dof_handler.n_dofs());
    m_concentration.reinit(dof_handler.n_dofs());
    m_tmp.reinit(dof_handler.n_dofs());
}

void Fraction::compute_derivatives(double)
{
    m_derivative = 0;

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
