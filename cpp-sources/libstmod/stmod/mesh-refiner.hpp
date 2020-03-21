#ifndef MESH_REFINER_HPP_INCLUDED
#define MESH_REFINER_HPP_INCLUDED

#include "stmod/fe-common.hpp"
#include "stmod/i-mesh-based.hpp"

#include <vector>

class MeshRefiner
{
public:
    MeshRefiner(FEResources& fe_res);
    void add_mesh_based(IMeshBased* object);
    void do_refine();

private:
    void pull_values();
    void push_values();
    void update_objects();
    void estimate();
    void refine_and_transfer();

    FEResources& m_fe_res;
    std::vector<IMeshBased*> m_objects;

    std::vector<dealii::Vector<double>> m_solutions_to_transfer;
    std::vector<dealii::Vector<double>> m_solutions_transferred;
};

#endif // MESH_REFINER_HPP_INCLUDED
