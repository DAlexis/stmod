#include "stmod/fractions.hpp"

void FractionsStorage::clear_subiteration()
{
    m_current_value = m_previous_value;
    for (auto &f : m_delta)
    {
        f = 0;
    }
}

void FractionsStorage::add_rhs_to_delta(double m)
{
    // m_delta += m_rhs * m;

    for (size_t i = 0; i < m_delta.size(); i++)
    {
        m_delta[i].add(m, m_rhs[i]);
    }
}

void FractionsStorage::make_sub_iteration(double dt)
{
    //m_current_value = m_previous_value + m_rhs * dt;

    m_current_value = m_previous_value;
    for (size_t i = 0; i < m_delta.size(); i++)
    {
        m_current_value[i].add(dt, m_rhs[i]);
    }
}

void FractionsStorage::step()
{
    /*
     * m_current_value = m_previous_value = m_previous_value + m_delta;
     * m_delta = 0.0;
     */

    for (size_t i = 0; i < m_delta.size(); i++)
    {
        m_previous_value[i].add(1.0, m_delta[i]);
    }
    clear_subiteration();
}

void FractionsStorage::collect_values(std::vector<double>& values) const
{
    for (auto &v : m_current_value)
    {
        for (dealii_vector_size_type i = 0; i < v.size(); i++) {
            values.push_back(v[i]);
        }
    }
}

void FractionsStorage::collect_deltas(std::vector<double>& deltas) const
{
    for (auto &d : m_delta)
    {
        for (dealii_vector_size_type i = 0; i < d.size(); i++) {
            deltas.push_back(d[i]);
        }
    }
}

void FractionsStorage::set_values(std::vector<double>::const_iterator& values)
{
    for (auto &v : m_delta)
    {
        for (dealii_vector_size_type i = 0; i < v.size(); i++) {
            v[i] = *(values++);
        }
    }
}
