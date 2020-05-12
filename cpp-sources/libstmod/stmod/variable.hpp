#ifndef VARIABLE_HPP_INCLUDED
#define VARIABLE_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class Variable
{
public:
    virtual dealii::Vector<double>& values_w() = 0;
    virtual const dealii::Vector<double>& values() const
    {
        return const_cast<Variable*>(this)->values_w();
    }

    virtual ~Variable() = default;
};

#endif // VARIABLE_HPP_INCLUDED
