#include "stmod/lin-eq-solver.hpp"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

using namespace dealii;

LinEqSolver::LinEqSolver(const dealii::AffineConstraints<double>& constraints, double precision) :
    m_constraints(constraints), m_precision(precision)
{
}

void LinEqSolver::solve(
        const dealii::SparseMatrix<double>& system_matrix,
        dealii::Vector<double>& solution,
        const dealii::Vector<double>& system_rhs,
        double precision_scale,
        const std::string& name) const
{
    std::cout << name << ": Solving linear equations... " << std::flush;
    SolverControl solver_control(m_iterations_count_max, m_precision * precision_scale);
    SolverCG<>    solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    m_constraints.distribute(solution);
    std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}
