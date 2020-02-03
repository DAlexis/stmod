#ifndef FRACTIONS_BASE_HPP_INCLUDED
#define FRACTIONS_BASE_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class IFractionData
{
public:
    virtual ~IFractionData() {}
    virtual const std::string& name() const = 0 ;
    virtual const dealii::Vector<double>& value() const = 0;
};


#endif // FRACTIONS_BASE_HPP_INCLUDED
