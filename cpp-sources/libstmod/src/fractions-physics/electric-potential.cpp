#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/matgen.hpp"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

using namespace dealii;

ElectricPotential::ElectricPotential(const FEResources& fe_res) :
    m_fe_res(fe_res)
{
}

void ElectricPotential::init(double needle_potential, double bottom_potential)
{
    m_needle_potential = needle_potential;
    m_bottom_potential = bottom_potential;

    m_solution.reinit(m_fe_res.dof_handler().n_dofs());
    m_system_rhs.reinit(m_fe_res.dof_handler().n_dofs());
    m_system_rhs2.reinit(m_fe_res.dof_handler().n_dofs());

    m_system_matrix.reinit(m_fe_res.sparsity_pattern());

    m_needle_potential_func = std::make_shared<ConstFunc>(needle_potential);
    m_bottom_potential_func = std::make_shared<ConstFunc>(bottom_potential);

    m_boundary_funcs.clear();
    m_boundary_funcs[1] = m_bottom_potential_func.get();
    m_boundary_funcs[2] = m_needle_potential_func.get();

    VectorTools::interpolate_boundary_values(m_fe_res.dof_handler(),
                                               1,
                                               Functions::ConstantFunction<2>(m_bottom_potential),
                                               m_boundary_values);
    VectorTools::interpolate_boundary_values(m_fe_res.dof_handler(),
                                               2,
                                               Functions::ConstantFunction<2>(m_needle_potential),
                                               m_boundary_values);

    m_system_matrix.copy_from(m_fe_res.r_laplace_matrix_axial());
}

void ElectricPotential::solve()
{
    assemble_system();
    solve_lin_eq();
}

void ElectricPotential::assemble_system()
{
    m_system_rhs = 0;
    //add_dirichlet_rhs_axial(m_fe_res.dof_handler(), m_system_rhs, m_boundary_funcs);

    MatrixTools::apply_boundary_values(m_boundary_values,
                                         m_system_matrix,
                                         m_solution,
                                         m_system_rhs);
    //m_system_rhs.print();
}

void ElectricPotential::solve_lin_eq()
{
    std::cout << "Potential: Solving linear equations" << std::endl;
    SolverControl solver_control(5000, 1e-2 * (fabs(m_bottom_potential) + fabs(m_needle_potential)));
    SolverCG<>        solver(solver_control);
    solver.solve(m_system_matrix, m_solution, m_system_rhs, PreconditionIdentity());
    std::cout << "Potential: " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    m_fe_res.constraints().distribute(m_solution);
}

const std::string& ElectricPotential::name() const
{
    return m_name;
}

const dealii::Vector<double>& ElectricPotential::value() const
{
    return m_solution;
}
