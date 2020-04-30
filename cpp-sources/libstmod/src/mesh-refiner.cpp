#include "stmod/mesh-refiner.hpp"

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

using namespace dealii;

MeshRefiner::MeshRefiner(FEGlobalResources& fe_global_res) :
    m_fe_global_res(fe_global_res)
{
}

void MeshRefiner::add_mesh_based(IMeshBased* object)
{
    m_objects.push_back(object);
}

void MeshRefiner::do_refine()
{
    pull_values();
    estimate();
    refine_and_transfer();
    update_objects();
    push_values();
}


void MeshRefiner::pull_values()
{
    m_solutions_to_transfer.resize(m_objects.size());
    for (size_t i = 0; i < m_objects.size(); i++)
    {
        m_solutions_to_transfer[i] = m_objects[i]->values_vector();
    }
}

void MeshRefiner::push_values()
{
    for (size_t i = 0; i < m_objects.size(); i++)
    {
        m_objects[i]->values_vector() = m_solutions_transferred[i];
    }
}

void MeshRefiner::update_objects()
{
    for (auto &object : m_objects)
    {
        object->init_mesh_dependent();
    }
}

void MeshRefiner::estimate()
{
    const unsigned int max_grid_level = 2;
    for (auto &object : m_objects)
    {
        Vector<float> estimated_error_per_cell(m_fe_global_res.triangulation().n_active_cells());

        KellyErrorEstimator<2>::estimate(
            m_fe_global_res.dof_handler(),
            QGauss<2 - 1>(m_fe_global_res.dof_handler().get_fe().degree + 1),
            std::map<types::boundary_id, const Function<2> *>(),
            object->error_estimation_vector(),
            estimated_error_per_cell
        );

        GridRefinement::refine_and_coarsen_fixed_fraction(m_fe_global_res.triangulation(),
                                                          estimated_error_per_cell,
                                                          0.7,
                                                          0.4, 150000);

        if (m_fe_global_res.triangulation().n_levels() > max_grid_level)
        {
            for (const auto &cell : m_fe_global_res.triangulation().active_cell_iterators_on_level(max_grid_level))
            {
                cell->clear_refine_flag();
            }
        }
        break;
        /*
        for (const auto &cell : triangulation.active_cell_iterators_on_level(min_grid_level))
        {
            cell->clear_coarsen_flag();
        }*/
    }
}

void MeshRefiner::refine_and_transfer()
{
    SolutionTransfer<2> soltution_transfer(m_fe_global_res.dof_handler());

    // prepare the triangulation,
    m_fe_global_res.triangulation().prepare_coarsening_and_refinement();

    // prepare the SolutionTransfer object for coarsening and refinement and give
    // the solution vector that we intend to interpolate later,
    soltution_transfer.prepare_for_coarsening_and_refinement(m_solutions_to_transfer);

    // actually execute the refinement,
    m_fe_global_res.triangulation().execute_coarsening_and_refinement();

    // Recreating all triangulation-related FE resources
    m_fe_global_res.on_triangulation_updated();

    const unsigned int new_n_dofs = m_fe_global_res.dof_handler().n_dofs();

    m_solutions_transferred.clear();
    m_solutions_transferred.resize(m_solutions_to_transfer.size());
    for (auto &it : m_solutions_transferred)
    {
        it.reinit(new_n_dofs);
    }

    // and interpolate the solution
    soltution_transfer.interpolate(m_solutions_to_transfer, m_solutions_transferred);

}
