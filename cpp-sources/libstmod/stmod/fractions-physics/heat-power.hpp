#ifndef HEAT_POWER_HPP_INCLUDED
#define HEAT_POWER_HPP_INCLUDED

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

#include <deal.II/lac/sparse_direct.h>

class HeatPower : public SecondaryValue
{
public:
    HeatPower(const FEGlobalResources& fe_res, const dealii::Vector<double>& concentration, const dealii::Vector<double>& potential, double mu);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepJob
    void compute(double t) override;

private:
    void create_rhs();

    void create_rhs_for_single_shape_func(dealii::Vector<double>& target, unsigned int shape_func_index);

    const FEGlobalResources& m_fe_global_res;
    const dealii::Vector<double>& m_concentration;
    const dealii::Vector<double>& m_potential;
    dealii::Vector<double> m_rhs;
    double m_mu;

};

#endif // HEAT_POWER_HPP_INCLUDED
