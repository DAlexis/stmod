#ifndef FRACTION_HPP_INCLUDED
#define FRACTION_HPP_INCLUDED

#include "stmod/grid/mesh-based.hpp"
#include "stmod/output/output-provider.hpp"
#include "stmod/time/time-iterable.hpp"

#include <deal.II/lac/vector.h>

enum class Sign
{
    plus = 1,
    minus = -1
};

class Fraction : public MeshBased, public IOutputProvider, public VariableWithDerivative
{
public:
    Fraction(const std::string& name);

    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    void compute_derivatives(double t) override;

    Fraction& add_source(double coeff, const dealii::Vector<double>& s1);
    Fraction& add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2);
    Fraction& add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2, const dealii::Vector<double>& s3);
    Fraction& add_source(double coeff, const dealii::Vector<double>& s1, const dealii::Vector<double>& s2, const dealii::Vector<double>& s3, const dealii::Vector<double>& s4);

    dealii::Vector<double>& values_w() override;

    const dealii::Vector<double>& derivatives() const;

    // IOutputProvider
    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    size_t output_values_count() const override;
    Fraction& operator=(double value);

protected:
    struct Source
    {
        double coeff = 1.0;
        std::vector<const dealii::Vector<double>*> sources;
    };

//    using Source = std::vector<const dealii::Vector<double>*>;
    void add_source_to_derivative(const Source& src);
    std::vector<Source> m_sources;
    //std::vector<double> m_signs;

    dealii::Vector<double> m_derivative;
    dealii::Vector<double> m_concentration;
    dealii::Vector<double> m_tmp;

    const std::string m_name;
};

class FractionWithImplicit : public Fraction
{
public:
    virtual const dealii::Vector<double>& get_implicit_delta(double dt, double theta = 0.5) = 0;
};

#endif // FRACTION_HPP_INCLUDED
