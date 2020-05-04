#ifndef I_STEPPABLE_HPP_INCLUDED
#define I_STEPPABLE_HPP_INCLUDED

#include "stmod/i-variables-storage.hpp"

#include <deal.II/lac/vector.h>

class ISteppable : public IVariablesStorage
{
public:
    virtual const dealii::Vector<double>& derivatives() const = 0;
    virtual void compute_derivetives(double t) = 0;

    virtual ~ISteppable() = default;
};

class IPreStepJob
{
public:
    virtual void compute(double t) = 0;

    virtual ~IPreStepJob() = default;
};

class IImplicitSteppable
{
public:
    virtual const dealii::Vector<double>& get_implicit_delta(double dt, double theta = 0.5) = 0;
    virtual ~IImplicitSteppable() = default;
};

#endif // I_STEPPABLE_HPP_INCLUDED
