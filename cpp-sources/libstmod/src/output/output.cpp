#include "stmod/output/output.hpp"

#include <deal.II/numerics/data_out.h>

#include <fstream>

void OutputMaker::add(const IOutputProvider* fraction_data)
{
    m_fractions.push_back(fraction_data);
}

void OutputMaker::output(const dealii::DoFHandler<2>& dof_handler, const std::string& filename)
{
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    for (auto &frac : m_fractions)
    {
        for (size_t i = 0; i < frac->output_values_count(); i++)
        {
            data_out.add_data_vector(frac->output_value(i), frac->output_name(i));
        }
    }
    data_out.build_patches();
    std::ofstream output(filename.c_str());
    data_out.write_vtu(output);
}
