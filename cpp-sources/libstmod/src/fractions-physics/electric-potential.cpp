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

using namespace dealii;

ElectricPotential::ElectricPotential(const FEGlobalResources& fe_res) :
    SecondaryValue("Potential"), m_fe_global_res(fe_res), m_electric_field_sampler(m_fe_global_res.dof_handler())
{
    //m_fe_res.set_boundary_cond_gen([this](auto & constraints) { add_boundary_conditions(constraints); });
}

void ElectricPotential::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);
    m_system_rhs.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_total_charge.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_E_scalar.reinit(m_fe_global_res.dof_handler().n_dofs());
    m_E_vector.resize(m_fe_global_res.dof_handler().n_dofs());

    m_system_matrix.reinit(m_fe_global_res.sparsity_pattern());

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
    std::cout << "Computing electric potential..." << std::endl;

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
    std::cout << "Computing electric potential done" << std::endl;
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

const std::vector<dealii::Point<2>>& ElectricPotential::E_vector()
{
    return m_E_vector;
}

const dealii::Vector<double>& ElectricPotential::E_scalar()
{
    return m_E_scalar;
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
    m_electric_field_sampler.sample(m_value, FESampler::Targets::grad_lap);
    auto & grads = m_electric_field_sampler.gradients();
    for (auto i = 0; i < grads.size(); i++)
    {
        auto & grad = grads[i];
        m_E_scalar[i] = grad.norm();
        m_E_vector[i] = -grad;
    }
}
