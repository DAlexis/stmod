#ifndef FRACTION_BASE_HPP_INCLUDED
#define FRACTION_BASE_HPP_INCLUDED

#include "stmod/i-mesh-based.hpp"
#include "stmod/i-output-provider.hpp"
#include "stmod/i-steppable.hpp"

#include <deal.II/lac/vector.h>

class Fraction : public IMeshBased, public IOutputProvider, public ISteppable
{
public:
    Fraction(const std::string& name);

    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    void compute_derivetives(double t) override;

    void add_single_source(double reaction_const, const dealii::Vector<double>& source);
    void add_pair_source(double reaction_const, const dealii::Vector<double>& source1, const dealii::Vector<double>& source2);

    const dealii::Vector<double>& concentration() const;
    dealii::Vector<double>& concentration_w();

    const dealii::Vector<double>& derivatives() const;

    // IOutputProvider
    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    size_t output_values_count() const override;

protected:
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    dealii::Vector<double> m_derivative;
    dealii::Vector<double> m_concentration;
    dealii::Vector<double> m_tmp;

    std::vector<const dealii::Vector<double>*> m_single_sources;
    std::vector<double> m_single_reaction_consts;

    std::vector<PairSourceTuple> m_pair_sources;
    std::vector<double> m_pair_reaction_consts;

    const std::string m_name;
};

#endif // FRACTION_BASE_HPP_INCLUDED
