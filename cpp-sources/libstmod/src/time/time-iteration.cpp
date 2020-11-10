#include "stmod/time/time-iteration.hpp"
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/numbers.h>

#include <tbb/parallel_for.h>
#include <stdexcept>

using namespace dealii;

VariablesCollector::VariablesCollector(const dealii::AffineConstraints<double>& constraints) :
    m_constraints(constraints)
{
}

void VariablesCollector::add_derivatives_provider(VariableWithDerivative* steppable)
{
    m_variables.push_back(steppable);
}

void VariablesCollector::add_pre_step_computator(IPreStepComputer* pre_step)
{
    m_pre_step_jobs.push_back(pre_step);
}

void VariablesCollector::add_implicit_steppable(ImplicitSteppable* implicit_steppable)
{
    m_implicit_steppables.push_back(implicit_steppable);
}

dealii::Vector<double>& VariablesCollector::stored_values()
{
    return m_values;
}

const dealii::Vector<double>& VariablesCollector::all_derivatives() const
{
    return m_derivatives;
}

void VariablesCollector::push_values(const Vector<double> &y)
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_variables)
    {
        auto & target_vector = steppable->values_w();
        copy_vector_part(target_vector, 0, target_vector.size(),
                         y, current_offset);
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

void VariablesCollector::pull_values_to_storage()
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_variables)
    {
        const auto & values = steppable->values();

        copy_vector_part(m_values, current_offset, values.size(),
                         values, 0);
        current_offset += values.size();
    }
}

void VariablesCollector::pull_derivatives()
{
    assert_size();
    dealii::Vector<double>::size_type current_offset = 0;
    for (auto steppable : m_variables)
    {
        const auto & derivatives = steppable->derivatives();

        copy_vector_part(m_derivatives, current_offset, derivatives.size(),
                         derivatives, 0);
        current_offset += derivatives.size();
    }
}

void VariablesCollector::compute_in_places(double t)
{
    //std::cout << "VariablesCollector::compute for t = " << t << std::endl;

    for (auto pre_step : m_pre_step_jobs)
    {
        pre_step->compute(t);
    }

    tbb::parallel_for(
        (size_t) 0, m_variables.size(),
        [this, t]( size_t l )
        {
            m_variables[l]->compute_derivatives(t);
        }
    );
/*
    for (auto steppable : m_variables)
    {
        steppable->compute_derivatives(t);
    }*/
}

void VariablesCollector::implicit_deltas_collect(double t, double dt, double theta)
{
    m_implicit_deltas.clear();
    for (auto& imp : m_implicit_steppables)
    {
        m_implicit_deltas.push_back(&imp->get_implicit_delta(dt, theta));
    }
}

void VariablesCollector::implicit_deltas_add()
{
    for (size_t i = 0; i < m_implicit_deltas.size(); i++)
    {
        m_implicit_steppables[i]->values_w().add(1.0, *m_implicit_deltas[i]);
    }
}

const dealii::Vector<double>& VariablesCollector::compute_derivatives(double t, const Vector<double> &y, double limiting_dt)
{
    push_values(y);
    compute_in_places(t);
    pull_derivatives();
    //limit_derivatives(limiting_dt, y, m_derivatives);
    return all_derivatives();
}

const VariableWithDerivative* VariablesCollector::variable_by_global_index(size_t index)
{
    if (m_variables.empty())
        return nullptr;

    return m_variables[index / m_variables[0]->values().size()];
}

void VariablesCollector::limit_derivatives(double dt, const dealii::Vector<double>& y, dealii::Vector<double>& derivatives)
{
    size_t count = 0;
    for (dealii::Vector<double>::size_type i = 0; i < y.size(); i++)
    {
        if (y[i] >= 0 && derivatives[i] < 0)
        {
            derivatives[i] = std::max(derivatives[i], -y[i] / dt);
            count++;
        }
    }
    std::cout << "Limiting applied to " << count << " of " << y.size() << " derivetives";
}

void VariablesCollector::assert_finite()
{
    for (dealii::Vector<double>::size_type i = 0; i < m_derivatives.size(); i++)
    {
        if ( !dealii::numbers::is_finite(m_derivatives[i]) )
        {
            // @todo
        }
    }
}

dealii::Vector<double>::size_type VariablesCollector::get_total_size()
{
    dealii::Vector<double>::size_type total_size = 0;
    for (auto steppable : m_variables)
    {
        total_size += steppable->values().size();
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

void SimpleTimeStepEstimator::min_x_over_dot_x(const dealii::Vector<double>& x, const dealii::Vector<double>& dot_x)
{
    fastest_time = 1e100;
    for (dealii::Vector<double>::size_type i = 0; i < x.size(); i++)
    {
        if (dot_x[i] != 0.0 && x[i] != 0.0)
        {
            double time = fabs(x[i] / dot_x[i]);
            if (time < fastest_time)
            {
                fastest_time = time;
                fastest_index = i;
            }
        }
    }
}

StmodTimeStepper::StmodTimeStepper()
{
}

void StmodTimeStepper::init()
{
    m_stepper = std::make_shared<TimeStepping::ExplicitRungeKutta<Vector<double>>>(TimeStepping::FORWARD_EULER);
    //m_stepper = std::make_shared<TimeStepping::ExplicitRungeKutta<Vector<double>>>(TimeStepping::RK_CLASSIC_FOURTH_ORDER);



    const double coarsen_param = 1.1;
    const double refine_param  = 0.9;
    const double min_delta     = 1e-13;
    const double max_delta     = 1e-11;

    /*const double refine_tol    = 1e-5;
    const double coarsen_tol   = 1e-7;*/

    const double refine_tol    = 1e-3;
    const double coarsen_tol   = 1e-5;

    m_embedded_stepper = std::make_shared<TimeStepping::EmbeddedExplicitRungeKutta<Vector<double>>>(
        //TimeStepping::FORWARD_EULER,
        TimeStepping::CASH_KARP,
        coarsen_param,
        refine_param,
        min_delta,
        max_delta,
        refine_tol,
        coarsen_tol
    );

    m_implicit_stepper = std::make_shared<TimeStepping::ImplicitRungeKutta<Vector<double>>>(TimeStepping::BACKWARD_EULER);
}


double StmodTimeStepper::iterate(VariablesCollector& collector, double t, double dt)
{
    init();
    std::cout << "=> Time step on t = " << t << std::endl;
    collector.resize();

    // Reading values to single array
    collector.pull_values_to_storage();

    // Saving this array
    m_on_explicit_begin = collector.stored_values();

    auto derivs = collector.compute_derivatives(t, m_on_explicit_begin, 0);
    SimpleTimeStepEstimator stse;
    stse.min_x_over_dot_x(m_on_explicit_begin, derivs);

    std::cout << "estimated steps: " << stse.fastest_time << "  for " << collector.variable_by_global_index(stse.fastest_index)->name() << " (index=" << stse.fastest_index << ") | ";

    double resulting_t = t;

    std::cout << "   Substeps: " << std::flush;
    // Making explicit time step
    if (collector.stored_values().size() != 0)
    {
        resulting_t = m_embedded_stepper->evolve_one_time_step(
        //resulting_t = m_stepper->evolve_one_time_step(
        //resulting_t = m_implicit_stepper->evolve_one_time_step(
            [&collector, dt](const double time, const Vector<double> &y)
            {
                std::cout << "t = " << time << " | " << std::flush;
                return collector.compute_derivatives(time, y, dt / 2);
            },
            t, dt, collector.stored_values()
        );

    } else {
        // @todo check this branch
        collector.compute_in_places(t);
        resulting_t += dt;
    }

    std::cout << "done" << std::endl;

    // Saving explicit time step results
    m_on_explicit_end = collector.stored_values();
    remove_negative(m_on_explicit_end);

    // Restoring old values to compute implicit part
    collector.push_values(m_on_explicit_begin);
    // Computing implicit deltas based on actual dt
    double actual_dt = resulting_t - t;

    std::cout << "   Implicit step on t = " << t << " with dt = " << actual_dt << "..." << std::endl;
    collector.implicit_deltas_collect(t, actual_dt, 0.5);

    // Restoring results of explicit step
    collector.push_values(m_on_explicit_end);
    // and adding deltas from implicit
    collector.implicit_deltas_add();

    collector.pull_values_to_storage();
    remove_negative(collector.stored_values());
    collector.push_values(collector.stored_values());

    std::cout << "   done, t changed from " << t << " to " << resulting_t << " with dt = " << actual_dt << "." << std::endl;
    return resulting_t;
}

void StmodTimeStepper::remove_negative(dealii::Vector<double>& values)
{
    size_t removed = 0;
    for (dealii::Vector<double>::size_type i = 0; i < values.size(); i++)
    {
        if (values[i] < 0.0)
        {
            values[i] = 0.0;
            removed++;
        }
    }
    std::cout << "Removed " << removed << " negative values of total " << values.size();
}
