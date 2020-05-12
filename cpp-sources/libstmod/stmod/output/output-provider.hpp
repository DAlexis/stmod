#ifndef I_OUTPUT_PROVIDER_HPP_INCLUDED
#define I_OUTPUT_PROVIDER_HPP_INCLUDED

#include <deal.II/lac/vector.h>

class IOutputProvider
{
public:
    virtual ~IOutputProvider() {}
    virtual const std::string& output_name(size_t index) const = 0 ;
    virtual const dealii::Vector<double>& output_value(size_t index) const = 0;
    virtual size_t output_values_count() const = 0;
};


#endif // I_OUTPUT_PROVIDER_HPP_INCLUDED
