#ifndef L2NORM_HPP
#define L2NORM_HPP

#include "stmod/fractions/secondary-value.hpp"

class L2Norm :  public SecondaryValue
{
public:
    L2Norm(const dealii::Vector<double>& field_x,
           const dealii::Vector<double>& field_y,
           const std::string& name);

    // IPreStepComputer
    void compute(double t) override;

private:
    const dealii::Vector<double>& m_field_x;
    const dealii::Vector<double>& m_field_y;
};

#endif // L2NORM_HPP
