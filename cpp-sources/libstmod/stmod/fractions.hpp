#ifndef FRACTIONS_HPP_INCLUDED
#define FRACTIONS_HPP_INCLUDED

#include "dsiterpp/integration.hpp"

#include <deal.II/lac/vector.h>

class FractionsStorage : public dsiterpp::IVariable
{
public:
    void clear_subiteration() override;
    void add_rhs_to_delta(double m) override;
    void make_sub_iteration(double dt) override;
    void step() override;
    void collect_values(std::vector<double>& values) const override;
    void collect_deltas(std::vector<double>& deltas) const override;
    void set_values(std::vector<double>::const_iterator& values) override;

private:
    std::vector<dealii::Vector<double>> m_previous_value;
    std::vector<dealii::Vector<double>> m_current_value;
    std::vector<dealii::Vector<double>> m_delta;
    std::vector<dealii::Vector<double>> m_rhs;

    using dealii_vector_size_type = dealii::Vector<double>::size_type;
};

#endif // FRACTIONS_HPP_INCLUDED
