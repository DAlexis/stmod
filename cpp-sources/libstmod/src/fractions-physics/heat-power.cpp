#include "stmod/fractions-physics/heat-power.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <tbb/tbb.h>

HeatPower::HeatPower(const FEGlobalResources& fe_res, const dealii::Vector<double>& concentration, const dealii::Vector<double>& potential, double mu) :
    SecondaryValue("heat_power_2"), m_fe_global_res(fe_res), m_concentration(concentration), m_potential(potential), m_mu(mu)
{
}


// IMeshBased
void HeatPower::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);
    m_rhs.reinit(dof_handler.n_dofs());
}

// IPreStepJob
void HeatPower::compute(double t)
{
    create_rhs();
    dealii::SparseDirectUMFPACK reverse_mass_matrix;
    reverse_mass_matrix.initialize(m_fe_global_res.mass_matrix());
    reverse_mass_matrix.vmult(m_value, m_rhs);
    m_fe_global_res.constraints().distribute(m_value);
}

void HeatPower::create_rhs()
{
    //std::cout << "HeatPower::create_rhs()" << std::endl;

    using namespace dealii;

    const unsigned int dofs_per_cell = m_fe_global_res.fe().dofs_per_cell;

    std::vector<dealii::Vector<double>> rhs_parts(dofs_per_cell);

    for (auto & rhs_part : rhs_parts)
    {
        rhs_part.reinit(m_rhs.size());
        rhs_part = 0;
    }
    tbb::parallel_for(
        (unsigned int) 0, dofs_per_cell,
        [this, &rhs_parts]( size_t l )
        {
            create_rhs_for_single_shape_func(rhs_parts[l], l);
        }
    );
    /*
    for (unsigned int l = 0; l < dofs_per_cell; ++l)
    {
        create_rhs_for_single_shape_func(rhs_parts[l], l);
    }*/

    m_rhs = 0;
    for (const auto & rhs_part : rhs_parts)
    {
        m_rhs += rhs_part;
    }

    m_rhs *= fabs(m_mu);
    //std::cout << "HeatPower::create_rhs() done" << std::endl;
}

void HeatPower::create_rhs_for_single_shape_func(dealii::Vector<double>& target, unsigned int shape_func_index)
{
    const auto l = shape_func_index;

    const double r_epsilon = 1e-4;

    using namespace dealii;

    auto &fe = m_fe_global_res.fe();

    const dealii::Quadrature<2> quadrature = dealii::QGauss<2>(4);

    FEValues<2, 2> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : m_fe_global_res.dof_handler().active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        // Iterating over quadrature points

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            double fe_values_shape_value_l_q_index = fe_values.shape_value(l, q_index);

            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                            double rhs_component_part = m_concentration[local_dof_indices[i]] * m_potential[local_dof_indices[j]] * m_potential[local_dof_indices[k]]
                                    * fe_values.shape_value(i, q_index)
                                    * (fe_values.shape_grad(j, q_index) * fe_values.shape_grad(k, q_index))
                                    * fe_values_shape_value_l_q_index
#ifndef DEBUG_NO_AXIAL
                                    * (r + r_epsilon)
#endif
                                    * fe_values.JxW(q_index);
                            target[local_dof_indices[l]] += rhs_component_part;
                    }
                }
            }
        }
    }
}
