#ifndef ELECTRIC_POTENTIAL_HPP_INCLUDED
#define ELECTRIC_POTENTIAL_HPP_INCLUDED

#include "stmod/fractions-base.hpp"
#include "stmod/fe-common.hpp"

class ElectricPotential : public IFractionData
{
public:
    ElectricPotential(const FEResources& fe_res);

    void init(double needle_potential = 7, double bottom_potential = -3);

    void solve();

    const std::string& name() const override;
    const dealii::Vector<double>& value() const override;

private:
    using ConstFunc = dealii::Functions::ConstantFunction<2>;
    void assemble_system();
    void solve_lin_eq();

    const FEResources& m_fe_res;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_system_rhs2;
    dealii::Vector<double> m_solution;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    double m_needle_potential = 0, m_bottom_potential = 0;
    std::shared_ptr<ConstFunc> m_needle_potential_func;
    std::shared_ptr<ConstFunc> m_bottom_potential_func;

    std::map<dealii::types::boundary_id, const dealii::Function<2>*> m_boundary_funcs;

    const std::string m_name = "Electric_potential";
};

#endif // ELECTRIC_POTENTIAL_HPP_INCLUDED
