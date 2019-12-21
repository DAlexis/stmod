#ifndef FIELD_OUTPUT_HPP_INCLUDED
#define FIELD_OUTPUT_HPP_INCLUDED

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>

#include <string>

class ScalarOutputMaker
{
public:
    ScalarOutputMaker(const dealii::DoFHandler<2>& dof_handler);

    void output_scalar(const dealii::Vector<double>& vector, const std::string& filename, const std::string& scalar_name = "scalar");

private:

    const dealii::DoFHandler<2>&  m_dof_handler;
};

class VectorOutputMaker
{
public:
    VectorOutputMaker(const dealii::Triangulation<2>& triangulation, unsigned int polynomial_degree);

    void set_vector(const std::vector<dealii::Point<2>>& values, const std::vector<std::string> component_names);

    void output(const std::string& filename);

private:
    const dealii::Triangulation<2>& m_triangulation;

    dealii::FESystem<2>      m_fe;
    dealii::Vector<double>   m_values;
    std::vector<std::string> m_component_names;

};

#endif // FIELD_OUTPUT_HPP_INCLUDED
