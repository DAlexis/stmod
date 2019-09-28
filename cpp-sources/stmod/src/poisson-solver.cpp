#include "include/poisson-solver.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
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

PoissonSolver::PoissonSolver(dealii::Triangulation<2>& initial_triangulation)
    : triangulation(initial_triangulation), fe(1), dof_handler(triangulation)
{}

void PoissonSolver::setup_system()
{
    dof_handler.distribute_dofs(fe);
    std::cout << "     Number of degrees of freedom: " << dof_handler.n_dofs()
                        << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

void PoissonSolver::assemble_system()
{
    QGauss<2> quadrature_formula(fe.degree + 1);
    QGauss<1> face_quadrature_formula(3);

    const RightHandSide right_hand_side;
    FEValues<2> fe_values(
        fe, quadrature_formula,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values
    );

    FEFaceValues<2> fe_face_values(
        fe, face_quadrature_formula,
        update_values         | update_quadrature_points  |
        update_normal_vectors | update_JxW_values
    );

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>         cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs        = 0;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto x_q = fe_values.quadrature_point(q_index);
                    const double r = x_q[0];
                    cell_matrix(i, j) += (
                        fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                        r *                                // r
                        fe_values.JxW(q_index)             // dx
                    );
                }
                const auto x_q = fe_values.quadrature_point(q_index);
                const double r = x_q[0];
                cell_rhs(i) += (
                    fe_values.shape_value(i, q_index) * // phi_i(x_q)
                    right_hand_side.value(x_q) *        // f(x_q)
                    r *                                 // r
                    fe_values.JxW(q_index)              // dx
                );
            }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i],
                                                    local_dof_indices[j],
                                                    cell_matrix(i, j));
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
/*
        // Neumann conditrions
        for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
        {
            if (!cell->face(face_number)->at_boundary())
                continue;
            fe_face_values.reinit(cell, face_number);
        }*/

    }

    double phi_0 = 0.123;
    double phi_L = 1.321;
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(
        dof_handler,
        1,
        Functions::ConstantFunction<2>(phi_0),
        //BoundaryValues<2>(),
        boundary_values
    );

    VectorTools::interpolate_boundary_values(
        dof_handler,
        2,
        Functions::ConstantFunction<2>(phi_L),
        //BoundaryValues<2>(),
        boundary_values
    );

    MatrixTools::apply_boundary_values(
        boundary_values,
        system_matrix,
        solution,
        system_rhs
    );
}

void PoissonSolver::solve_lin_eq()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>        solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "     " << solver_control.last_step()
                        << " CG iterations needed to obtain convergence." << std::endl;
}

void PoissonSolver::output(const std::string& filename) const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
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

