#include "include/field-output.hpp"

#include <deal.II/fe/fe_q.h>
#include <fstream>

using namespace dealii;

VectorOutputMaker::VectorOutputMaker(const dealii::Triangulation<2>& triangulation, unsigned int polynomial_degree) :
    m_triangulation(triangulation), m_fe(FE_Q<2>(polynomial_degree), 2)
{
}

void VectorOutputMaker::set_vector(const std::vector<dealii::Point<2>>& values, const std::vector<std::string> component_names)
{
    m_component_names.clear();
    m_component_names.push_back(component_names[0]);
    m_component_names.push_back(component_names[1]);
    m_values.reinit(values.size() * 2);

    for (unsigned int i = 0; i< values.size(); i++)
    {
        m_values[2*i] = values[i][0];
        m_values[2*i + 1] = values[i][1];
    }
}

void VectorOutputMaker::output(const std::string& filename)
{
    dealii::DoFHandler<2> dof_handler(m_triangulation);
    dof_handler.distribute_dofs (m_fe);

    dealii::DataOut<2> data_out;

    data_out.attach_dof_handler (dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation (2, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector (m_values, m_component_names, DataOut<2>::type_dof_data, data_component_interpretation);
    data_out.build_patches();

    std::ofstream f_output(filename);
    data_out.write_vtk(f_output);
}
