#ifndef FRACTIONS_HPP_INCLUDED
#define FRACTIONS_HPP_INCLUDED

#include "dsiterpp/integration.hpp"

#include <deal.II/lac/vector.h>
#include <functional>

using SolutionInterpolatorFunc = std::function< std::vector<dealii::Vector<double>> (const std::vector<dealii::Vector<double>>& orig_solution) >;

class FractionsStorage : public dsiterpp::IVariable
{
public:
    FractionsStorage();

    void clear_subiteration() override;
    void add_rhs_to_delta(double m) override;
    void make_sub_iteration(double dt) override;
    void step() override;
    void collect_values(std::vector<double>& values) const override;
    void collect_deltas(std::vector<double>& deltas) const override;
    void set_values(std::vector<double>::const_iterator& values) override;

    void create_arrays(size_t fractions_count, unsigned int dimension);

    void resize_interpolate(SolutionInterpolatorFunc interpolator);

    void set_const_values(size_t fraction_index, double value);
    const dealii::Vector<double>& current(size_t fraction_index);
    const dealii::Vector<double>& previous(size_t fraction_index);
    dealii::Vector<double>& previous_w(size_t fraction_index);
    dealii::Vector<double>& rhs(size_t fraction_index);

private:
    void reinit_additional_arrays();

    std::vector<dealii::Vector<double>> m_previous_value;
    std::vector<dealii::Vector<double>> m_current_value;
    std::vector<dealii::Vector<double>> m_delta;
    std::vector<dealii::Vector<double>> m_rhs;

    using dealii_vector_size_type = dealii::Vector<double>::size_type;
};

class FractionBase : public dsiterpp::IRHS
{
public:
    FractionBase(FractionsStorage& storage, size_t fraction_index);

    size_t fraction_index();

protected:
    FractionsStorage& m_storage;
    size_t m_fraction_index;
};

#endif // FRACTIONS_HPP_INCLUDED
