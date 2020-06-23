#include "stmod/fractions-physics/heat-power.hpp"

HeatPower::HeatPower(const FEGlobalResources& fe_res, const dealii::Vector<double> concentration, const dealii::Vector<double> potential, double mu) :
    SecondaryValue("heat_power"), m_fe_global_res(fe_res), m_concentration(concentration), m_potential(potential), m_mu(mu)
{
}


// IMeshBased
void HeatPower::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
}

// IPreStepJob
void HeatPower::compute(double t)
{
}
