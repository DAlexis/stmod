#include "stmod/fractions/fraction.hpp"


Fraction::Fraction(const std::string& name) :
    Variable(name)
{
}

Fraction& Fraction::add_source(double coeff, const dealii::Vector<double>& s1)
{
    Source s;
    s.coeff = coeff;
    s.sources = {&s1};
    m_sources.push_back(s);
    return *this;
}

Fraction& Fraction::add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2)
{
    Source s;
    s.coeff = coeff;
    s.sources = {&s1, &s2};
    m_sources.push_back(s);
    return *this;
}

Fraction& Fraction::add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2, const dealii::Vector<double>& s3)
{
    Source s;
    s.coeff = coeff;
    s.sources = {&s1, &s2, &s3};
    m_sources.push_back(s);
    return *this;
}

Fraction& Fraction::add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2, const dealii::Vector<double>& s3, const dealii::Vector<double>& s4)
{
    Source s;
    s.coeff = coeff;
    s.sources = {&s1, &s2, &s3, &s4};
    m_sources.push_back(s);
    return *this;
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

Fraction& Fraction::operator=(double value)
{
    m_concentration = value;
    return *this;
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
    for (auto & s : m_sources)
    {
        add_source_to_derivative(s);
    }
}

void Fraction::add_source_to_derivative(const Source& src)
{
    m_tmp = *src.sources[0];
    m_tmp *= src.coeff;
    for (size_t i = 1; i < src.sources.size(); i++)
    {
        m_tmp.scale(*src.sources[i]);
    }
    m_derivative += m_tmp;
}
