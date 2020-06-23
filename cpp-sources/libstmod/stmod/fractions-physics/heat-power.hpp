#ifndef HEAT_POWER_HPP_INCLUDED
#define HEAT_POWER_HPP_INCLUDED

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

#include <deal.II/lac/sparse_direct.h>

class HeatPower : public SecondaryValue
{
public:
    HeatPower(const FEGlobalResources& fe_res, const dealii::Vector<double> concentration, const dealii::Vector<double> potential, double mu);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepJob
    void compute(double t) override;

private:
    const FEGlobalResources& m_fe_global_res;
    const dealii::Vector<double> m_concentration;
    const dealii::Vector<double> m_potential;
    double m_mu;

};

#endif // HEAT_POWER_HPP_INCLUDED
