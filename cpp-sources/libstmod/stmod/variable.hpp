#ifndef VARIABLE_HPP_INCLUDED
#define VARIABLE_HPP_INCLUDED

#include <deal.II/lac/vector.h>

#include <string>

class ScalarVariable
{
public:
    ScalarVariable(const std::string& name) : m_name(name) { }
    virtual dealii::Vector<double>& values_w() = 0;
    virtual const dealii::Vector<double>& values() const
    {
        return const_cast<ScalarVariable*>(this)->values_w();
    }

    virtual ~ScalarVariable() = default;

    const std::string& name() const { return m_name; }

    double operator[](dealii::types::global_dof_index i) const { return values()[i]; }

    double& operator[](dealii::types::global_dof_index i) { return values_w()[i]; }

    operator const dealii::Vector<double>& () { return values(); }

    ScalarVariable& operator=(double value) { values_w() = value; return *this; }

protected:
    const std::string m_name;
};

#endif // VARIABLE_HPP_INCLUDED
