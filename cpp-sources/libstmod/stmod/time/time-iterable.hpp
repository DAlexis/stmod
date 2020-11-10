#ifndef I_STEPPABLE_HPP_INCLUDED
#define I_STEPPABLE_HPP_INCLUDED

#include "stmod/variable.hpp"

#include <deal.II/lac/vector.h>

class VariableWithDerivative : public virtual ScalarVariable
{
public:
    virtual const dealii::Vector<double>& derivatives() const = 0;
    virtual void compute_derivatives(double t) = 0;

    virtual ~VariableWithDerivative() = default;
};

class IPreStepComputer
{
public:
    virtual void compute(double t) = 0;

    virtual ~IPreStepComputer() = default;
};

class ImplicitSteppable : public virtual ScalarVariable
{
public:
    virtual const dealii::Vector<double>& get_implicit_delta(double dt, double theta = 0.5) = 0;
    virtual ~ImplicitSteppable() = default;
};

#endif // I_STEPPABLE_HPP_INCLUDED
