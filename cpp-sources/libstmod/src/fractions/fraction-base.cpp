#include "stmod/fractions/secondary-value.hpp"

SecondaryValue::SecondaryValue(const std::string& name) :
    m_name(name)
{
}

const dealii::Vector<double>& SecondaryValue::error_estimation_vector() const
{
    return m_value;
}

void SecondaryValue::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    m_value.reinit(dof_handler.n_dofs());
}

dealii::Vector<double>& SecondaryValue::values_w()
{
    return m_value;
}

const std::string& SecondaryValue::output_name(size_t index) const
{
    return m_name;
}

const dealii::Vector<double>& SecondaryValue::output_value(size_t index) const
{
    return m_value;
}

size_t SecondaryValue::output_values_count() const
{
    return 1;
}

SecondaryConstant::SecondaryConstant(const std::string& name, double value) :
    SecondaryValue(name), m_current_scalar_value(value)
{

}

void SecondaryConstant::compute(double)
{
    m_value = m_current_scalar_value;
}

SecondaryConstant& SecondaryConstant::operator=(double value)
{
    m_current_scalar_value = value;
    m_value = m_current_scalar_value;
    return *this;
}

SecondaryConstant::operator double() const
{
    return m_current_scalar_value;
}

SecondaryFunction::SecondaryFunction(const std::string& name, Lambda func) :
    SecondaryValue(name), m_func(func)
{
}

void SecondaryFunction::compute(double t)
{
    for (dealii::types::global_dof_index i = 0; i < m_value.size(); i++)
    {
        m_value[i] = m_func(i, t);
    }
}
