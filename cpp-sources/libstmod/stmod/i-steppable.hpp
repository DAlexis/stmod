#ifndef I_STEPPABLE_HPP_INCLUDED
#define I_STEPPABLE_HPP_INCLUDED

#include "stmod/i-variables-storage.hpp"

#include <deal.II/lac/vector.h>

class ISteppable : public IVariablesStorage
{
public:
    virtual const dealii::Vector<double>& derivatives_vector() const = 0;
    virtual void compute(double t) = 0;

    virtual ~ISteppable() = default;
};

class IPreStepJob
{
public:
    virtual void compute(double t) = 0;

    virtual ~IPreStepJob() = default;
};

#endif // I_STEPPABLE_HPP_INCLUDED
