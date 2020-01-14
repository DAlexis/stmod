#include "stmod/fractions.hpp"

FractionsStorage::FractionsStorage()
{

}

void FractionsStorage::clear_subiteration()
{
    for (size_t i = 0; i < m_current_value.size(); i++)
    {
        m_current_value[i] = m_previous_value[i];
    }

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

void FractionsStorage::resize_interpolate(SolutionInterpolatorFunc interpolator)
{
    m_previous_value = interpolator(m_previous_value);
    reinit_additional_arrays();
    clear_subiteration();
}

void FractionsStorage::set_const_values(size_t fraction_index, double value)
{
    for (unsigned int i = 0; i < m_previous_value[fraction_index].size(); i++)
    {
        m_previous_value[fraction_index][i] = value;
    }
}

const dealii::Vector<double>& FractionsStorage::current(size_t fraction_index)
{
    return m_current_value[fraction_index];
}

const dealii::Vector<double>& FractionsStorage::previous(size_t fraction_index)
{
    return m_previous_value[fraction_index];
}

dealii::Vector<double>& FractionsStorage::previous_w(size_t fraction_index)
{
    return m_previous_value[fraction_index];
}

dealii::Vector<double>& FractionsStorage::rhs(size_t fraction_index)
{
    return m_rhs[fraction_index];
}

void FractionsStorage::create_arrays(size_t fractions_count, unsigned int dimension)
{
    m_previous_value.clear();
    m_current_value.clear();
    m_delta.clear();
    m_rhs.clear();

    for (size_t i=0; i < fractions_count; i++)
    {
        m_previous_value.push_back(dealii::Vector<double>(dimension));
        m_current_value.push_back(dealii::Vector<double>(dimension));
        m_delta.push_back(dealii::Vector<double>(dimension));
        m_rhs.push_back(dealii::Vector<double>(dimension));
    }
}

void FractionsStorage::reinit_additional_arrays()
{
    unsigned int new_dim = m_previous_value[0].size();
    for (size_t i = 0; i < m_previous_value.size(); i++)
    {
        m_current_value[i].reinit(new_dim);
        m_delta[i].reinit(new_dim);
        m_rhs[i].reinit(new_dim);
    }
}


FractionRHSBase::FractionRHSBase(FractionsStorage& storage, size_t fraction_index) :
    m_storage(storage), m_fraction_index(fraction_index)
{
}

size_t FractionRHSBase::fraction_index()
{
    return m_fraction_index;
}
