#ifndef FULL_MODELS_MODEL_ONE_HPP_INCLUDED
#define FULL_MODELS_MODEL_ONE_HPP_INCLUDED

#include "stmod/grid/grid.hpp"
#include "stmod/output/output.hpp"

class FEGlobalResources;
class MeshRefiner;

class ElectricPotential;
class Electrons;
class VariablesCollector;

class ModelOne
{
public:
    ModelOne();
    ~ModelOne();

    void init_grid();
    void init_fractions();
    void assign_initial_values();

    void run();

private:
    Grid m_grid;
    OutputMaker m_output_maker;

    std::unique_ptr<BoundaryAssigner> m_boundary_assigner;
    std::unique_ptr<FEGlobalResources> m_global_resources;
    std::unique_ptr<MeshRefiner> m_refiner;
    std::unique_ptr<VariablesCollector> m_variables_collector;

    std::unique_ptr<ElectricPotential> m_electric_potential;
    std::unique_ptr<Electrons> m_electrons;
};

#endif // FULL_MODELS_MODEL_ONE_HPP_INCLUDED
