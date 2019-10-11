#ifndef __VECTOR_OUTPUT_HPP__
#define __VECTOR_OUTPUT_HPP__

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>

#include <string>

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

#endif // __VECTOR_OUTPUT_HPP__
