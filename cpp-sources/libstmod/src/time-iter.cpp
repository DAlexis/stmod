#include "stmod/time-iter.hpp"

StmodTimeStepper::StmodTimeStepper() :
    m_explicit_runge_kutta_stepper(dealii::TimeStepping::FORWARD_EULER)
{
}
