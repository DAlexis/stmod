#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

class Gradient : public SecondaryValue
{
public:
    Gradient(
            const FEGlobalResources& fe_res,
            const dealii::Vector<double>& field,
            const std::string& name,
            size_t component,
            double multiplier = 1.0);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepComputer
    void compute(double t) override;

private:
    const FEGlobalResources& m_fe_global_res;

    const dealii::Vector<double>& m_field;

    dealii::Vector<double> m_rhs;
    const size_t m_component;
    double m_multiplier;

};

#endif // GRADIENT_HPP
