#ifndef ELECTRIC_POTENTIAL_HPP_INCLUDED
#define ELECTRIC_POTENTIAL_HPP_INCLUDED

#include "stmod/output-provider.hpp"
#include "stmod/steppable.hpp"
#include "stmod/fe-common.hpp"

class ElectricPotential : public IOutputProvider
{
public:
    ElectricPotential(const FEResources& fe_res);

    void init(double needle_potential = 7, double bottom_potential = 2);

    void solve();

    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    virtual size_t output_values_count() const override;

    void add_charge(const dealii::Vector<double>& charge_vector, double mul = 1.0);

    const dealii::Vector<double>& total_chagre() const;

private:
    using ConstFunc = dealii::Functions::ConstantFunction<2>;

    void calc_total_charge();
    void assemble_system();
    void solve_lin_eq();

    const FEResources& m_fe_res;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_solution;

    std::vector<const dealii::Vector<double>*> m_charges;
    std::vector<double> m_charges_muls;
    dealii::Vector<double> m_total_charge;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    double m_needle_potential = 0, m_bottom_potential = 0;
    std::shared_ptr<ConstFunc> m_needle_potential_func;
    std::shared_ptr<ConstFunc> m_bottom_potential_func;

    std::map<dealii::types::boundary_id, const dealii::Function<2>*> m_boundary_funcs;

    const std::string m_name = "Electric_potential";
};

#endif // ELECTRIC_POTENTIAL_HPP_INCLUDED
