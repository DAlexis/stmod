#ifndef FRACTIONS_BASE_HPP_INCLUDED
#define FRACTIONS_BASE_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class IFractionData
{
public:
    virtual ~IFractionData() {}
    virtual const std::string& name(size_t index) const = 0 ;
    virtual const dealii::Vector<double>& value(size_t index) const = 0;
    virtual size_t values_count() const = 0;
};


#endif // FRACTIONS_BASE_HPP_INCLUDED
