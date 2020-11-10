#include  "stmod/fractions/l2norm.hpp"

L2Norm::L2Norm(const dealii::Vector<double>& field_x,
       const dealii::Vector<double>& field_y,
       const std::string& name) :
    m_field_x(field_x), m_field_y(field_y), ScalarVariable(name)
{
}

void L2Norm::compute(double)
{
    for (dealii::Vector<double>::size_type i = 0; i < m_value.size(); i++)
    {
        m_value[i] = sqrt(pow(m_field_x[i], 2) + pow(m_field_x[i], 2));
    }
}
