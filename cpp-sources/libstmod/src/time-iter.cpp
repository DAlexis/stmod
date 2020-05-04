#include "stmod/time-iter.hpp"
#include <deal.II/lac/affine_constraints.h>

#include <stdexcept>

using namespace dealii;

VariablesCollector::VariablesCollector(const dealii::AffineConstraints<double>& constraints) :
    m_constraints(constraints)
{
}

void VariablesCollector::add_steppable(ISteppable* steppable)
{
    m_steppables.push_back(steppable);
}

void VariablesCollector::add_pre_step(IPreStepJob* pre_step)
{
    m_pre_step_jobs.push_back(pre_step);
}

dealii::Vector<double>& VariablesCollector::all_values()
{
    return m_values;
}

const dealii::Vector<double>& VariablesCollector::all_derivatives() const
{
    return m_derivatives;
}

void VariablesCollector::push_values()
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_steppables)
    {
        auto & target_vector = steppable->values_w();
        copy_vector_part(target_vector, 0, target_vector.size(),
                         m_values, current_offset);
        current_offset += target_vector.size();

        m_constraints.distribute(target_vector);
    }
}

void VariablesCollector::resize()
{
    auto size = get_total_size();
    m_values.reinit(size);
    m_derivatives.reinit(size);
}

void VariablesCollector::pull_values()
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_steppables)
    {
        const auto & values = steppable->values_w();

        copy_vector_part(m_values, current_offset, values.size(),
                         values, 0);
        current_offset += values.size();
    }
}

void VariablesCollector::pull_derivatives()
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_steppables)
    {
        const auto & derivatives = steppable->derivatives();

        copy_vector_part(m_derivatives, current_offset, derivatives.size(),
                         derivatives, 0);
        current_offset += derivatives.size();
    }
}

void VariablesCollector::compute(double t)
{
    std::cout << "VariablesCollector::compute for t = " << t << std::endl;

    for (auto pre_step : m_pre_step_jobs)
    {
        pre_step->compute(t);
    }

    for (auto steppable : m_steppables)
    {
        steppable->compute_derivetives(t);
    }
}

dealii::Vector<double>::size_type VariablesCollector::get_total_size()
{
    dealii::Vector<double>::size_type total_size = 0;
    for (auto steppable : m_steppables)
    {
        total_size += steppable->values_w().size();
    }
    return total_size;
}

void VariablesCollector::assert_size()
{
    if (get_total_size() != m_values.size())
        throw std::runtime_error("VariablesCollector::pull_values(): m_values or m_derivatives vector size != total variables count");
}

void VariablesCollector::copy_vector_part(
        dealii::Vector<double>& target, dealii::Vector<double>::size_type target_begin, dealii::Vector<double>::size_type size,
        const dealii::Vector<double>& source, dealii::Vector<double>::size_type source_begin
        )
{
    memcpy(target.data() + target_begin, source.data() + source_begin, size * sizeof(double));
}

StmodTimeStepper::StmodTimeStepper()
{
}

void StmodTimeStepper::init()
{
    //m_stepper = std::make_shared<TimeStepping::ExplicitRungeKutta<Vector<double>>>(TimeStepping::FORWARD_EULER);
    m_stepper = std::make_shared<TimeStepping::ExplicitRungeKutta<Vector<double>>>(TimeStepping::RK_CLASSIC_FOURTH_ORDER);



    const double coarsen_param = 1.2;
    const double refine_param  = 0.8;
    const double min_delta     = 1e-11;
    const double max_delta     = 1e-7;

    /*const double refine_tol    = 1e-5;
    const double coarsen_tol   = 1e-7;*/

    const double refine_tol    = 1e-2;
    const double coarsen_tol   = 1e-4;

    m_embedded_stepper = std::make_shared<TimeStepping::EmbeddedExplicitRungeKutta<Vector<double>>>(
        TimeStepping::BOGACKI_SHAMPINE,
        coarsen_param,
        refine_param,
        min_delta,
        max_delta,
        refine_tol,
        coarsen_tol
    );
}


double StmodTimeStepper::iterate(VariablesCollector& collector, double t, double dt)
{
    collector.resize();
    collector.pull_values();
/*
    // Trivial Euler variant
    collector.compute(t);
    collector.pull_derivatives();
    collector.all_values().add(0.00000005, collector.all_derivatives());*/

    double resulting_t = m_embedded_stepper->evolve_one_time_step(
        [&collector](const double time, const Vector<double> &y)
        {
            collector.push_values();
            collector.compute(time);
            collector.pull_derivatives();
            return collector.all_derivatives();
        },
        t,
        dt,
        collector.all_values()
    );
    collector.push_values();
    return resulting_t;
}

