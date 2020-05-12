#ifndef FRACTION_BASE_HPP_INCLUDED
#define FRACTION_BASE_HPP_INCLUDED

#include "stmod/grid/mesh-based.hpp"
#include "stmod/output/output-provider.hpp"
#include "stmod/time/time-iterable.hpp"

#include <deal.II/lac/vector.h>

class SecondaryValue : public MeshBased, public IOutputProvider, public IPreStepComputer
{
public:
    SecondaryValue(const std::string& name);

    const dealii::Vector<double>& error_estimation_vector() const;
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;
    dealii::Vector<double>& values_w() override;

    // IOutputProvider
    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    size_t output_values_count() const override;

protected:
    dealii::Vector<double> m_value;
    const std::string m_name;
};

class SecondaryConstant : public SecondaryValue
{
public:
    SecondaryConstant(const std::string& name, double values = 0);

    void compute(double t);

    SecondaryConstant& operator=(double values);
    operator double() const;

private:
    double m_current_scalar_value;
};

class SecondaryFunction : public SecondaryValue
{
public:
    using Lambda = std::function<double(dealii::types::global_dof_index, double)>;

    SecondaryFunction(const std::string& name, Lambda func);

    void compute(double t);

private:
    Lambda m_func;
};

#endif // FRACTION_BASE_HPP_INCLUDED
