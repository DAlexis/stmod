#ifndef I_MESH_BASED_HPP_INCLUDED
#define I_MESH_BASED_HPP_INCLUDED

#include "stmod/variable.hpp"

#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>

class MeshBased : virtual public ScalarVariable
{
public:
    virtual void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) = 0;
};

#endif // I_MESH_BASED_HPP_INCLUDED
