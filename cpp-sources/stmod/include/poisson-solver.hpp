#ifndef __POISSON_SOLVER_HPP__
#define __POISSON_SOLVER_HPP__

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <string>

class PoissonSolver
{
public:
    PoissonSolver(dealii::Triangulation<2>& initial_triangulation);

    void solve();
    void output(const std::string& filename) const;
    void refine_grid();

private:
    void setup_system();
    void assemble_system();
    void solve_lin_eq();


    dealii::Triangulation<2>& triangulation;
    dealii::AffineConstraints<double> constraints;

    dealii::FE_Q<2>                    fe;
    dealii::DoFHandler<2>        dof_handler;
    dealii::SparsityPattern            sparsity_pattern;
    dealii::SparseMatrix<double> system_matrix;
    dealii::Vector<double> solution;
    dealii::Vector<double> system_rhs;

    double phi_0 = 0;
    double pli_L = 1;
};

#endif // __POISSON_SOLVER_HPP__
