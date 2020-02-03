#ifndef MESH_OUTPUT_HPP_INCLUDED
#define MESH_OUTPUT_HPP_INCLUDED

#include "stmod/fractions-base.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <string>

class FractionsOutputMaker
{
public:
    void add(const IFractionData* fraction_data);
    void output(const dealii::DoFHandler<2>& dof_handler, const std::string& filename);

private:
    std::vector<const IFractionData*> m_fractions;
};

#endif // MESH_OUTPUT_HPP_INCLUDED
