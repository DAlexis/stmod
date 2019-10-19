#include "include/poisson-solver.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

using namespace dealii;


class RightHandSide : public Function<2>
{
public:
    RightHandSide()
        : Function<2>()
    {}
    virtual double value(const Point<2> & p,
                                             const unsigned int component = 0) const override;
};

class BoundaryValues : public Function<2>
{
public:
    BoundaryValues()
        : Function<2>()
    {}
    virtual double value(const Point<2> & p,
                                             const unsigned int component = 0) const override;
};

double RightHandSide::value(const Point<2> &p,
                                                                 const unsigned int /*component*/) const
{
    return 0.0;
/*    double return_value = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        return_value += 4.0 * std::pow(p(i), 4.0);
    return return_value;*/
}

double BoundaryValues::value(const Point<2> &p,
                                                                    const unsigned int /*component*/) const
{
    return p.square();
}

PoissonSolver::PoissonSolver(dealii::Triangulation<2>& initial_triangulation, unsigned int polynomial_degree)
    : m_triangulation(initial_triangulation), m_fe(polynomial_degree), m_dof_handler(m_triangulation)
{}

void PoissonSolver::setup_system()
{
    m_dof_handler.distribute_dofs(m_fe);
    m_solution.reinit(m_dof_handler.n_dofs());
    m_system_rhs.reinit(m_dof_handler.n_dofs());
    constraints.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler, constraints);

    double phi_0 = 0.123;
    double phi_L = 10.321;

    VectorTools::interpolate_boundary_values(
        m_dof_handler,
        1,
        Functions::ConstantFunction<2>(phi_0),
        constraints
    );

    VectorTools::interpolate_boundary_values(
        m_dof_handler,
        2,
        Functions::ConstantFunction<2>(phi_L),
        constraints
    );

    constraints.close();
    DynamicSparsityPattern dsp(m_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
    m_sparsity_pattern.copy_from(dsp);
    m_system_matrix.reinit(m_sparsity_pattern);
}

void PoissonSolver::assemble_system()
{
    const RightHandSide right_hand_side;

    const QGauss<2> quadrature_formula(m_fe.degree + 1);
    FEValues<2> fe_values(m_fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = m_fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double r = x_q[0];
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += (
                        fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                        r *                                // r
                        fe_values.JxW(q_index)             // dx
                    );
                }
                cell_rhs(i) += (
                  fe_values.shape_value(i, q_index) * // phi_i(x_q)
                  right_hand_side.value(x_q) *        // f(x_q)
                  r *                                 // r
                  fe_values.JxW(q_index)              // dx
                );
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, m_system_matrix, m_system_rhs);
    }
}

void PoissonSolver::solve_lin_eq()
{
    SolverControl solver_control(4000, 1e-12);
    SolverCG<>        solver(solver_control);
    solver.solve(m_system_matrix, m_solution, m_system_rhs, PreconditionIdentity());
    std::cout << "     " << solver_control.last_step()
                        << " CG iterations needed to obtain convergence." << std::endl;
    constraints.distribute(m_solution);
}

void PoissonSolver::estimate_error()
{
    m_estimated_error_per_cell.reinit(m_triangulation.n_active_cells());
    KellyErrorEstimator<2>::estimate(
        m_dof_handler,
        QGauss<2 - 1>(m_fe.degree + 1),
        std::map<types::boundary_id, const Function<2> *>(),
        m_solution,
        m_estimated_error_per_cell);
}

const std::vector<dealii::Vector<double>>& PoissonSolver::refine_and_coarsen_grid(const std::vector<dealii::Vector<double>>& solutions_to_interpolate)
{
    const size_t solutions_count = solutions_to_interpolate.size();

    dealii::SolutionTransfer<2> soltution_transfer(m_dof_handler);
    // flag some cells for refinement and coarsening, e.g.
    GridRefinement::refine_and_coarsen_fixed_number(m_triangulation,
                                                      m_estimated_error_per_cell,
                                                      0.3,
                                                      0.03);
    // prepare the triangulation,
    m_triangulation.prepare_coarsening_and_refinement();

    // prepare the SolutionTransfer object for coarsening and refinement and give
    // the solution vector that we intend to interpolate later,

    std::vector<dealii::Vector<double>> qq;
    soltution_transfer.prepare_for_coarsening_and_refinement(solutions_to_interpolate);

    // actually execute the refinement,
    m_triangulation.execute_coarsening_and_refinement();

    // redistribute dofs,
    m_dof_handler.distribute_dofs(m_fe);
    const unsigned int new_n_dofs = m_dof_handler.n_dofs();

    m_solutions_interpolated.clear();
    m_solutions_interpolated.resize(solutions_count);
    for (auto &it : m_solutions_interpolated)
    {
        it.reinit(new_n_dofs);
    }

    // and interpolate the solution
    soltution_transfer.interpolate(solutions_to_interpolate, m_solutions_interpolated);
    return m_solutions_interpolated;
}


dealii::Triangulation<2>& PoissonSolver::triangulation()
{
    return m_triangulation;
}

const dealii::DoFHandler<2>& PoissonSolver::dof_handler() const
{
    return m_dof_handler;
}

const dealii::Vector<double> PoissonSolver::solution()
{
    return m_solution;
}

unsigned int PoissonSolver::polynomial_degree()
{
    return m_fe.degree;
}

void PoissonSolver::output(const std::string& filename) const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler(m_dof_handler);
    data_out.add_data_vector(m_solution, "solution");
    data_out.build_patches();
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

void PoissonSolver::solve()
{
    std::cout << "Solving problem in 2 space dimensions." << std::endl;
    setup_system();
    assemble_system();
    solve_lin_eq();
}

