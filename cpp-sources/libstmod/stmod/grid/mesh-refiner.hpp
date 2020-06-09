#ifndef MESH_REFINER_HPP_INCLUDED
#define MESH_REFINER_HPP_INCLUDED

#include "stmod/fe-common.hpp"
#include "stmod/grid/mesh-based.hpp"

#include <vector>

class MeshRefiner
{
public:
    MeshRefiner(FEGlobalResources& fe_global_res);
    void add_mesh_based(MeshBased* object);
    void do_refine();
    void call_on_mesh_refine();

private:
    void pull_values();
    void push_values();
    void update_objects();
    void estimate();
    void refine_and_transfer();

    FEGlobalResources& m_fe_global_res;
    std::vector<MeshBased*> m_objects;

    std::vector<dealii::Vector<double>> m_solutions_to_transfer;
    std::vector<dealii::Vector<double>> m_solutions_transferred;
};

#endif // MESH_REFINER_HPP_INCLUDED
