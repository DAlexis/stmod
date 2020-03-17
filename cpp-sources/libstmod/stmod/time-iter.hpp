#ifndef TIME_ITER_HPP_INCLUDED
#define TIME_ITER_HPP_INCLUDED

#include <deal.II/base/time_stepping.h>

class StmodTimeStepper
{
public:
    StmodTimeStepper();

private:
    dealii::TimeStepping::ExplicitRungeKutta<dealii::Vector<double>> m_explicit_runge_kutta_stepper;
};

#endif // TIME_ITER_HPP_INCLUDED
