#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/matgen.hpp"
#include "stmod/phys-consts.hpp"
#include "stmod/matgen.hpp"
#include "stmod/grid/grid.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

const std::string ElectricPotential::m_name = "Electric_potential";

ElectricPotential::ElectricPotential(const FEGlobalResources& fe_res) :
    Variable(m_name), m_fe_global_res(fe_res), m_electric_field_sampler(m_fe_global_res.dof_handler())
{
    //m_fe_res.set_boundary_cond_gen([this](auto & constraints) { add_boundary_conditions(constraints); });
}

void ElectricPotential::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);
    m_system_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_total_charge.reinit(m_fe_global_res.dof_handler().n_dofs());

    m_E_vector.resize(m_fe_global_res.dof_handler().n_dofs());

    m_Ex_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_Ey_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());

    m_E_scalar.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_E_x.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_E_y.reinit(m_fe_global_res.dof_handler().n_dofs());

    m_system_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_E_x_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_E_y_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());

    m_mass_matrix_inverse.initialize(m_fe_global_res.mass_matrix());
    create_r_grad_phi_i_comp_phi_j_axial(m_fe_global_res.dof_handler(), m_E_x_rhs_matrix, 0);
    create_r_grad_phi_i_comp_phi_j_axial(m_fe_global_res.dof_handler(), m_E_y_rhs_matrix, 1);

    // Creating bounndary values map
    m_boundary_values.clear();
    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::top_and_needle,
                                             Functions::ConstantFunction<2>(m_electric_parameters.needle_potential),
                                             m_boundary_values);

    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::bottom,
                                             Functions::ConstantFunction<2>(m_electric_parameters.bottom_potential),
                                             m_boundary_values);
}

void ElectricPotential::compute(double t)
{
    //std::cout << "Computing electric potential..." << std::endl;

    m_total_charge = 0;
    for (size_t i=0; i < m_charges.size(); i++)
    {
        m_total_charge.add(m_charges_muls[i], *(m_charges[i]));
    }

    m_system_rhs = 0;
    m_fe_global_res.mass_matrix().vmult(m_system_rhs, m_total_charge);
    m_system_rhs *= 1 / Consts::epsilon_0;

    m_system_matrix.copy_from(m_fe_global_res.laplace_matrix());
    m_fe_global_res.constraints().condense(m_system_matrix, m_system_rhs);

    MatrixTools::apply_boundary_values(m_boundary_values,
                                     m_system_matrix,
                                     m_value,
                                     m_system_rhs);

    m_system_matrix_inverse.initialize(m_system_matrix);
    m_system_matrix_inverse.vmult(m_value, m_system_rhs);

    m_fe_global_res.constraints().distribute(m_value);

    create_e_field();
    //std::cout << "Computing electric potential done" << std::endl;
}

void ElectricPotential::add_charge(const dealii::Vector<double>& charge_vector, double mul)
{
    m_charges.push_back(&charge_vector);
    m_charges_muls.push_back(mul);
}

const dealii::Vector<double>& ElectricPotential::total_chagre() const
{
    return m_total_charge;
}

void ElectricPotential::set_electric_parameters(const ElectricParameters& electric_parameters)
{
    m_electric_parameters = electric_parameters;
}

const std::vector<Tensor<1, 2>>& ElectricPotential::E_vector()
{
    return m_E_vector;
}

const dealii::Vector<double>& ElectricPotential::E_scalar()
{
    return m_E_scalar;
}

const dealii::Vector<double>& ElectricPotential::E_x()
{
    return m_E_x;
}

const dealii::Vector<double>& ElectricPotential::E_y()
{
    return m_E_y;
}


void ElectricPotential::calc_total_charge()
{
    m_total_charge = 0;
    for (size_t i=0; i < m_charges.size(); i++)
    {
        m_total_charge.add(m_charges_muls[i], *(m_charges[i]));
    }
}

void ElectricPotential::create_e_field()
{
    m_E_x = 0;
    m_E_y = 0;

    m_E_x_rhs_matrix.vmult(m_Ex_rhs, m_value);
    m_mass_matrix_inverse.vmult(m_E_x, m_Ex_rhs);
    m_fe_global_res.constraints().distribute(m_Ex_rhs);

    m_E_y_rhs_matrix.vmult(m_Ey_rhs, m_value);
    m_mass_matrix_inverse.vmult(m_E_y, m_Ey_rhs);
    m_fe_global_res.constraints().distribute(m_Ey_rhs);

    for (unsigned int i = 0; i < m_E_scalar.size(); i++)
    {
        m_E_scalar[i] = sqrt(pow(m_E_x[i], 2) + pow(m_E_y[i], 2));
    }

    /*
    m_E_vector.resize(m_fe_global_res.n_dofs());
    m_E_scalar.reinit(m_fe_global_res.n_dofs());

    const dealii::Quadrature<2> & support_points = m_fe_global_res.fe().get_generalized_support_points();
    FEValues<2> fe_values(m_fe_global_res.fe(), support_points, update_gradients);

    const unsigned int dofs_per_cell = m_fe_global_res.fe().dofs_per_cell;
    const unsigned int n_support_points  = support_points.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : m_fe_global_res.dof_handler().active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        // target_support_point_index -- index of support point where we need to calculate gradient
        for (unsigned int target_support_point_index = 0; target_support_point_index < n_support_points; ++target_support_point_index)
        {
            Tensor<1, 2> gradient_value;
            for (unsigned int i = 0; i < dofs_per_cell; ++i) // Iterating over all base functions in this cell
            {
                const unsigned int current_dof_index = local_dof_indices[i];
                const double current_coefficient = m_value[current_dof_index];
                gradient_value += current_coefficient * fe_values.shape_grad(target_support_point_index, i);
            }
            m_E_vector[local_dof_indices[target_support_point_index]] = -gradient_value;
        }
    }

    for (types::global_dof_index i = 0; i < m_E_vector.size(); i++)
    {
        m_E_scalar[i] = m_E_vector[i].norm();
    }*/

    /*
    m_electric_field_sampler.sample(m_value, FESampler::Targets::grad_lap);
    auto & grads = m_electric_field_sampler.gradients();
    for (auto i = 0; i < grads.size(); i++)
    {
        auto & grad = grads[i];
        m_E_scalar[i] = grad.norm();
        m_E_vector[i] = -grad;
    }*/

}

const std::string& ElectricPotential::output_name(size_t index) const
{
    switch (index) {
    case 0:
    {
        return m_name_pot;
    }
    case 1:
    {
        return m_name_Ex;
    }
    case 2:
    {
        return m_name_Ey;
    }
    case 3:
    {
        return m_name_E;
    }

    default:
        throw std::invalid_argument("Invalid output index");
    }
}

const dealii::Vector<double>& ElectricPotential::output_value(size_t index) const
{
    switch (index) {
    case 0:
        return m_value;
    case 1:
        return m_E_x;
    case 2:
        return m_E_y;
    case 3:
        return m_E_scalar;
    default:
        throw std::invalid_argument("Invalid output index");
    }
}

size_t ElectricPotential::output_values_count() const
{
    return 4;
}
