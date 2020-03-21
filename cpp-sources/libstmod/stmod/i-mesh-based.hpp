#ifndef I_MESH_BASED_HPP_INCLUDED
#define I_MESH_BASED_HPP_INCLUDED

#include "stmod/i-variables-storage.hpp"

#include <deal.II/lac/vector.h>

class IMeshBased : public IVariablesStorage
{
public:
    virtual void init_mesh_dependent() = 0;

    virtual const dealii::Vector<double>& error_estimation_vector() const = 0;
};

#endif // I_MESH_BASED_HPP_INCLUDED
