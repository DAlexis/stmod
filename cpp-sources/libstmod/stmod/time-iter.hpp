#ifndef TIME_ITER_HPP_INCLUDED
#define TIME_ITER_HPP_INCLUDED

#include "stmod/i-steppable.hpp"

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/vector.h>

namespace dealii {
    template <typename T> class AffineConstraints;
}

class VariablesCollector
{
public:
    VariablesCollector(const dealii::AffineConstraints<double>& constraints);
    void add_steppable(ISteppable* steppable);
    void add_pre_step(IPreStepJob* pre_step);

    dealii::Vector<double>& all_values();
    const dealii::Vector<double>& all_derivatives() const;

    void push_values();

    void resize();
    void pull_values();
    void pull_derivatives();

    void compute(double t);

private:
    const dealii::AffineConstraints<double>& m_constraints;

    dealii::Vector<double>::size_type get_total_size();

    std::vector<ISteppable*> m_steppables;
    std::vector<IPreStepJob*> m_pre_step_jobs;

    dealii::Vector<double> m_values;
    dealii::Vector<double> m_derivatives;

    void assert_size();

    static void copy_vector_part(
            dealii::Vector<double>& target, dealii::Vector<double>::size_type target_begin, dealii::Vector<double>::size_type size,
            const dealii::Vector<double>& source, dealii::Vector<double>::size_type source_begin
            );
};

class StmodTimeStepper
{
public:
    StmodTimeStepper();
    void init();

    double iterate(VariablesCollector& collector, double t, double dt);

private:
    std::shared_ptr<dealii::TimeStepping::ExplicitRungeKutta<dealii::Vector<double>>> m_stepper;
    std::shared_ptr<dealii::TimeStepping::EmbeddedExplicitRungeKutta<dealii::Vector<double>>> m_embedded_stepper;
    //dealii::TimeStepping::EmbeddedExplicitRungeKutta<dealii::Vector<double>> m_explicit_runge_kutta_stepper;
};

#endif // TIME_ITER_HPP_INCLUDED
