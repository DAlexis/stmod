#ifndef STEPPABLE_HPP_INCLUDED
#define STEPPABLE_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class ISteppable
{
public:
    virtual dealii::Vector<double>& values_vector() = 0;
    virtual const dealii::Vector<double>& derivatives_vector() const = 0;
    virtual void compute(double t) = 0;
    virtual ~ISteppable() = default;
};

#endif // STEPPABLE_HPP_INCLUDED
