#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/matgen.hpp"
#include "stmod/phys-consts.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

using namespace dealii;

ElectricPotential::ElectricPotential(const FEResources& fe_res) :
    m_fe_res(fe_res)
{
}

dealii::Vector<double>& ElectricPotential::values_vector()
{
    return m_solution;
}

const dealii::Vector<double>& ElectricPotential::error_estimation_vector() const
{
    return m_solution;
}

void ElectricPotential::init_mesh_dependent()
{
    m_solution.reinit(m_fe_res.dof_handler().n_dofs());
    m_system_rhs.reinit(m_fe_res.dof_handler().n_dofs());
    m_system_rhs_boundary.reinit(m_fe_res.dof_handler().n_dofs());
    m_total_charge.reinit(m_fe_res.dof_handler().n_dofs());

    create_system_matrix_and_inverse_matrix();
}

void ElectricPotential::compute(double t)
{
    std::cout << "Computing electric potential..." << std::endl;
    calc_total_charge();
    create_rhs();
    solve_lin_eq();
    std::cout << "Computing electric potential done" << std::endl;
}

void ElectricPotential::create_rhs()
{
    m_system_rhs = 0;
    m_mass_matrix.vmult(m_system_rhs, m_total_charge);
    m_system_rhs *= - Consts::e / Consts::epsilon_0;

    //m_system_rhs.print();
    // Removing values at boundary conditions position
    m_system_rhs.scale(m_system_rhs_boundary_pattern);


    // Adding boundary conditions
    m_system_rhs += m_system_rhs_boundary;
    //m_system_rhs = m_system_rhs_boundary;
}

void ElectricPotential::solve_lin_eq()
{
    m_system_matrix_inverse.vmult(m_solution, m_system_rhs);
    /*
    m_fe_res.lin_eq_solver().solve(
                m_system_matrix, m_solution, m_system_rhs,
                (fabs(m_bottom_potential) + fabs(m_needle_potential))*0.000005, m_name);*/
/*
    std::cout << "Potential: Solving linear equations" << std::endl;
    SolverControl solver_control(5000, 1e-10);
    SolverCG<>        solver(solver_control);
    solver.solve(m_system_matrix, m_solution, m_system_rhs, PreconditionIdentity());
    std::cout << "Potential: " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    */
    m_potential_constraints.distribute(m_solution);
}

const std::string& ElectricPotential::output_name(size_t) const
{
    return m_name;
}

const dealii::Vector<double>& ElectricPotential::output_value(size_t) const
{
    return m_solution;
}

size_t ElectricPotential::output_values_count() const
{
    return 1;
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
    create_system_matrix_and_inverse_matrix();
}

void ElectricPotential::create_system_matrix_and_inverse_matrix()
{
    m_potential_constraints.clear();
        DoFTools::make_hanging_node_constraints(m_fe_res.dof_handler(), m_potential_constraints);
        VectorTools::interpolate_boundary_values(m_fe_res.dof_handler(),
                                               2,
                                               Functions::ConstantFunction<2>(m_electric_parameters.bottom_potential),
                                               m_potential_constraints);

        VectorTools::interpolate_boundary_values(m_fe_res.dof_handler(),
                                               1,
                                               Functions::ConstantFunction<2>(m_electric_parameters.needle_potential),
                                               m_potential_constraints);
    m_potential_constraints.close();

    DynamicSparsityPattern dsp(m_fe_res.dof_handler().n_dofs());
    DoFTools::make_sparsity_pattern(m_fe_res.dof_handler(),
                                  dsp,
                                  m_potential_constraints,
                                  /*keep_constrained_dofs = */ false);

    m_sparsity_pattern.copy_from(dsp);
    m_system_matrix.reinit(m_sparsity_pattern);
    m_mass_matrix.reinit(m_sparsity_pattern);

    create_r_laplace_matrix_axial(m_fe_res.dof_handler(),
                               m_system_matrix,
                               m_system_rhs_boundary,
                               m_potential_constraints,
                               QGauss<2>(2 * m_fe_res.dof_handler().get_fe().degree + 1));

    create_r_mass_matrix_axial(m_fe_res.dof_handler(),
                               m_mass_matrix,
                               m_potential_constraints,
                               QGauss<2>(2 * m_fe_res.dof_handler().get_fe().degree + 1));

    m_system_matrix_inverse.initialize(m_system_matrix);
    create_boundary_pattern();
}

void ElectricPotential::create_boundary_pattern()
{
    m_system_rhs_boundary_pattern.reinit(m_system_rhs_boundary.size());
    for (dealii::Vector<double>::size_type i = 0; i < m_system_rhs_boundary.size(); i++)
    {
        m_system_rhs_boundary_pattern[i] = (m_system_rhs_boundary[i] == 0.0) ? 1.0 : 0.0;
    }
}


void ElectricPotential::calc_total_charge()
{
    m_total_charge = 0;
    for (size_t i=0; i < m_charges.size(); i++)
    {
        m_total_charge.add(m_charges_muls[i], *(m_charges[i]));
    }
}
