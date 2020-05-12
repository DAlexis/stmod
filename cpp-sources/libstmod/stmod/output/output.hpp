#ifndef MESH_OUTPUT_HPP_INCLUDED
#define MESH_OUTPUT_HPP_INCLUDED

#include "stmod/output/output-provider.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <string>

class OutputMaker
{
public:
    void add(const IOutputProvider* fraction_data);
    void output(const dealii::DoFHandler<2>& dof_handler, const std::string& filename);

private:
    std::vector<const IOutputProvider*> m_fractions;
};

#endif // MESH_OUTPUT_HPP_INCLUDED
