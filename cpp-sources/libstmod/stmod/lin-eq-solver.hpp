#ifndef LIN_EQ_SOLVER_HPP_INCLUDED
#define LIN_EQ_SOLVER_HPP_INCLUDED

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <string>

class LinEqSolver
{
public:
    LinEqSolver(const dealii::AffineConstraints<double>& constraints, double precision = 1e-2);

    void solve(
            const dealii::SparseMatrix<double>& system_matrix,
            dealii::Vector<double>& solution,
            const dealii::Vector<double>& system_rhs,
            double precision_scale = 1.0,
            const std::string& name = "unknown variable") const;

public:
    const dealii::AffineConstraints<double>& m_constraints;
    double m_precision;
    unsigned int m_iterations_count_max = 15000;
};

#endif // LIN_EQ_SOLVER_HPP_INCLUDED
