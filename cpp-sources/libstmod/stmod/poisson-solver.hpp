#ifndef __POISSON_SOLVER_HPP__
#define __POISSON_SOLVER_HPP__

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/solution_transfer.h>

#include <string>

class SolutionsStorage
{
public:

private:
};

class PoissonSolver
{
public:
    PoissonSolver(dealii::Triangulation<2>& initial_triangulation, unsigned int polynomial_degree = 2);

    void solve();
    void output(const std::string& filename) const;
    void estimate_error();
    const std::vector<dealii::Vector<double>>& refine_and_coarsen_grid(const std::vector<dealii::Vector<double>>& solutions_to_interpolate);
    dealii::Triangulation<2>& triangulation();
    const dealii::DoFHandler<2>& dof_handler() const;

    const dealii::Vector<double>& solution();

    unsigned int polynomial_degree();

private:
    void setup_system();
    void assemble_system();
    void solve_lin_eq();


    dealii::Triangulation<2>& m_triangulation;
    dealii::AffineConstraints<double> constraints;

    dealii::FE_Q<2>        m_fe;
    dealii::DoFHandler<2>  m_dof_handler;
    dealii::SparsityPattern  m_sparsity_pattern;
    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_solution;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<float>  m_estimated_error_per_cell;

    std::vector<dealii::Vector<double>> m_solutions_interpolated;

    double phi_0 = 0;
    double pli_L = 1;
};

#endif // __POISSON_SOLVER_HPP__
