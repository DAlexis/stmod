#ifndef I_VARIABLES_PROVIDER_HPP_INCLUDED
#define I_VARIABLES_PROVIDER_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class IVariablesStorage
{
public:
    virtual dealii::Vector<double>& values_w() = 0;

    virtual ~IVariablesStorage() = default;
};

#endif // I_VARIABLES_PROVIDER_HPP_INCLUDED
