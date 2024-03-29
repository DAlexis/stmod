#ifndef TIME_ITER_HPP_INCLUDED
#define TIME_ITER_HPP_INCLUDED

#include "stmod/time/time-iterable.hpp"

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/vector.h>

namespace dealii {
    template <typename T> class AffineConstraints;
}

class VariablesCollector
{
public:
    VariablesCollector(const dealii::AffineConstraints<double>& constraints);
    void add_derivatives_provider(VariableWithDerivative* steppable);
    void add_pre_step_computator(IPreStepComputer* pre_step);
    void add_implicit_steppable(ImplicitSteppable* implicit_steppable);

    dealii::Vector<double>& stored_values();
    const dealii::Vector<double>& all_derivatives() const;

    void resize();

    /**
     * @brief Push specified values to fractions. Internal storage is unchanged
     * @param y - array to push
     */
    void push_values(const dealii::Vector<double> &y);

    /**
     * @brief Pull values from fractions to internal storage
     */
    void pull_values_to_storage();
    void pull_derivatives();

    void compute_in_places(double t);

    void implicit_deltas_collect(double t, double dt, double theta = 0.5);
    void implicit_deltas_add();

    /**
     * @brief Push specified values, compute and pull derivatives into internal storage
     * @param t - time
     * @param y - point where to compute
     * @return derivatives vector (reference to internal derivatives storage)
     */
    const dealii::Vector<double>& compute_derivatives(double t, const dealii::Vector<double> &y, double limiting_dt);

    const VariableWithDerivative* variable_by_global_index(size_t index);
private:
    static void limit_derivatives(double dt, const dealii::Vector<double>& y, dealii::Vector<double>& derivatives);
    void assert_finite(); /// @todo Function not ready!

    const dealii::AffineConstraints<double>& m_constraints;

    dealii::Vector<double>::size_type get_total_size();

    std::vector<VariableWithDerivative*> m_variables;
    std::vector<IPreStepComputer*> m_pre_step_jobs;
    std::vector<ImplicitSteppable*> m_implicit_steppables;
    std::vector<const dealii::Vector<double>*> m_implicit_deltas;

    dealii::Vector<double> m_values;
    dealii::Vector<double> m_derivatives;

    void assert_size();

    static void copy_vector_part(
            dealii::Vector<double>& target, dealii::Vector<double>::size_type target_begin, dealii::Vector<double>::size_type size,
            const dealii::Vector<double>& source, dealii::Vector<double>::size_type source_begin
            );
};

class SimpleTimeStepEstimator
{
public:
    void min_x_over_dot_x(const dealii::Vector<double>& x, const dealii::Vector<double>& dot_x);

    double fastest_time = 0.0;
    size_t fastest_index = 0;
};

class StmodTimeStepper
{
public:
    StmodTimeStepper();
    void init();

    double iterate(VariablesCollector& collector, double t, double dt);

private:
    static void remove_negative(dealii::Vector<double>& values);

    SimpleTimeStepEstimator m_estimator;

    std::shared_ptr<dealii::TimeStepping::ImplicitRungeKutta<dealii::Vector<double>>> m_implicit_stepper;
    std::shared_ptr<dealii::TimeStepping::ExplicitRungeKutta<dealii::Vector<double>>> m_stepper;
    std::shared_ptr<dealii::TimeStepping::EmbeddedExplicitRungeKutta<dealii::Vector<double>>> m_embedded_stepper;

    dealii::Vector<double> m_on_explicit_begin;
    dealii::Vector<double> m_on_explicit_end;
    //dealii::TimeStepping::EmbeddedExplicitRungeKutta<dealii::Vector<double>> m_explicit_runge_kutta_stepper;
};

#endif // TIME_ITER_HPP_INCLUDED
