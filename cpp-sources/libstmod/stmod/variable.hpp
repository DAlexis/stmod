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

    double operator[](dealii::types::global_dof_index i) const { return values()[i]; }

    double& operator[](dealii::types::global_dof_index i) { return values_w()[i]; }

    operator const dealii::Vector<double>& () { return values(); }

    Variable& operator=(double value) { values_w() = value; return *this; }
};

#endif // VARIABLE_HPP_INCLUDED
