#include "stmod/full-models/model-one.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/poisson-grid.hpp"

ModelOne::ModelOne()
{
    init_grid();
}

void ModelOne::run()
{
}

void ModelOne::init_grid()
{
    m_grid = std::make_shared<Grid>();
    m_grid->load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    BoundaryAssigner boundary_assigner(*m_grid);
    boundary_assigner.assign_boundary_ids();
}

void ModelOne::init_poisson_solver()
{
}
