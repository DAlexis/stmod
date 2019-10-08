#include <utility>

/*
 *  Copyright (C) 2017 IAPRAS - All Rights Reserved
 *
 *  This file is part of the GEC calculator.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "solver.h"

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_system.h>

#include <fstream>
#include <algorithm>

#include "crutches/util.hpp"
#include "crutches/log.hpp"

#include "gec_mesh_generator.h"
#include "grid_in_out.h"
#include "util.h"
#include "geometry.h"


using namespace dealii;


SolverParallel::SolverParallel(function_sigma sigma):
    quad_n_points(2),
    mpi_communicator(MPI_COMM_WORLD),
    this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator)),
    do_save_matrix(false),
    do_save_rhs(false),
    triangulation(mpi_communicator,
                  typename dealii::Triangulation<3>::MeshSmoothing
                  (dealii::Triangulation<3>::smoothing_on_refinement |
                   dealii::Triangulation<3>::smoothing_on_coarsening)),
    dof_handler(triangulation),
    fe(1),
    pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),
    get_sigma(std::move(sigma))
{}


SolverParallel::~SolverParallel()
{
    dof_handler.clear();
}


void SolverParallel::solve_problem(function_j_ext j_ext)
{
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells()
          << std::endl
          << "   Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    assemble_system(get_sigma, std::move(j_ext));

    solve_system();

    // computing_timer.print_summary();
    // computing_timer.reset();
}


void SolverParallel::create_grid(double _r0, double _r1, unsigned int refine_count)
{
    r0 = _r0;
    r1 = _r1;

    const Point<3> center (0, 0, 0);
    GridGenerator::hyper_shell(triangulation, center, r0, r1, 96);

    static const dealii::SphericalManifold<3> manifold_description(center);
    triangulation.set_manifold (0, manifold_description);
    triangulation.set_all_manifold_ids(0);
    for (unsigned int step = 0; step < refine_count; ++ step)
    {
        Triangulation::active_cell_iterator cell = triangulation.begin_active();
        Triangulation::active_cell_iterator endc = triangulation.end();
        for (; cell!=endc; ++cell)
        {
            cell->set_refine_flag();
        }
        triangulation.execute_coarsening_and_refinement();
    }

    mark_boundary();
}


void SolverParallel::create_grid_from_ini(const char *ini_file_name)
{
    double _r0 = 0, _r1 = 0;
    earth_surface_from_ini(triangulation, _r0, _r1, ini_file_name);
    r0 = _r0;
    r1 = _r1;

    mark_boundary();
}


void SolverParallel::create_grid_from_parameters(double _r0, double _r1,
                                                 size_t refine_number, size_t lower_z_levels,
                                                 double lower_z_height)
{
    r0 = _r0;
    r1 = _r1;

    std::vector<double> altitudes;

    earth_surface(triangulation,
                  _r0, _r1, refine_number, lower_z_levels, lower_z_height, altitudes);

    if (get_mpi_pid()) {
        std::ofstream f("altitudes.txt");
        f.precision(10);
        for (auto z: altitudes) {
            f << z << std::endl;
        }
        f.close();
    }

    mark_boundary();
}


void SolverParallel::load_grid(double _r0, double _r1, const char *grid_file_name,
                               const char *data_file_name)
{
    r0 = _r0;
    r1 = _r1;

    /*
    GridIn<3> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file(grid_file_name);

    if (ends_with(grid_file_name, ".ucd")) {
        grid_in.read_ucd(input_file);
    } else if (ends_with(grid_file_name, ".vtk")) {
        grid_in.read_vtk(input_file);
    }
     */

    /*
    std::ifstream input_file(grid_file_name);
    custom_read_ucd(triangulation, input_file, false);
    input_file.close();
     */

    {
        read_ucb(triangulation, grid_file_name);
    }

    triangulation.load(data_file_name);

    mark_boundary();
}

void SolverParallel::save_test_grid()
{
    {
        dealii::GridOut grid_out;
        std::ofstream of("mesh.vtk");
        grid_out.write_vtk(triangulation, of);
        of.close();
    }

    {
        dealii::GridOut grid_out;
        std::ofstream of("mesh.ucd");
        grid_out.write_ucd(triangulation, of);
        of.close();
    }

    {
        write_ucb(triangulation, "mesh.ucb");
    }

    triangulation.save("mesh.data");
}

void SolverParallel::mark_boundary()
{
    auto endc = triangulation.end();
    for (auto cell=triangulation.begin(); cell!=endc; ++cell)
    {
        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            if (cell->face(face_number)->at_boundary())
            {
                auto center = cell->face(face_number)->vertex(0);
                if (std::fabs(center.norm()-r0) < 1)
                {
                    cell->face(face_number)->set_boundary_id(0);
                }
                else if (std::fabs(center.norm()-r1) < 1)
                {
                    cell->face(face_number)->set_boundary_id(1);
                }
                else
                {
                    std::cout
                        << "Face " << center << " not on boundary 0 neither 1" << "\t"
                        << std::fabs(center.norm()-r0) << "\t"
                        << std::fabs(center.norm()-r1)
                        << std::endl;
                }
            }
        }
    }
}


void SolverParallel::save_triangulation(const char *triangulation_name)
{
    dealii::GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(triangulation, triangulation_name);
}


void SolverParallel::save_eps_triangulation(const char *eps_filename)
{
    std::ofstream out(eps_filename);
    GridOut grid_out;
    grid_out.write_eps(triangulation, out);
}


void SolverParallel::setup_system(const dealii::Function<3, double> &upper_boundary_function)
{
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    std::cout << "Sizes of vectors " << get_mpi_pid() << "\t"
              << locally_owned_dofs.size() << "\t"
              << locally_relevant_dofs.size() << std::endl;

    locally_relevant_solution_j_ext.reinit(locally_owned_dofs,
                                           locally_relevant_dofs, mpi_communicator);

    locally_relevant_solution_b_0_1.reinit(locally_owned_dofs,
                                           locally_relevant_dofs, mpi_communicator);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs, mpi_communicator);

    system_j_ext_rhs.reinit(locally_owned_dofs, mpi_communicator);
    system_b_0_1_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints_j_ext.clear();
    constraints_j_ext.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_j_ext);

    constraints_b_0_1.clear();
    constraints_b_0_1.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_b_0_1);

    Functions::ZeroFunction<3> bf_zero;
    Functions::ConstantFunction<3> bf_one(1.0);

    {
        std::map<types::boundary_id, const Function<3, double>*> boundary_map_1 =
        {
            {0, &bf_zero},
            {1, &upper_boundary_function}
        };

        VectorTools::interpolate_boundary_values(dof_handler, boundary_map_1, constraints_j_ext);
        constraints_j_ext.close();
    }

    {
        std::map<types::boundary_id, const Function<3, double>*> boundary_map_2 =
        {
            {0, &bf_zero},
            {1, &bf_one}
        };

        VectorTools::interpolate_boundary_values(dof_handler, boundary_map_2, constraints_b_0_1);
        constraints_b_0_1.close();
    }

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp,
                                    constraints_b_0_1, false);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.n_locally_owned_dofs_per_processor(),
                                               mpi_communicator, locally_relevant_dofs);

    system_j_ext_matrix.reinit(locally_owned_dofs, locally_owned_dofs,
                         dsp, mpi_communicator);
    system_b_0_1_matrix.reinit(locally_owned_dofs, locally_owned_dofs,
                         dsp, mpi_communicator);

    /*
    {
        SparsityPattern sparsity_pattern;
        sparsity_pattern.copy_from(dsp);
        std::ofstream out("sparsity_pattern1.svg");
        sparsity_pattern.print_svg(out);
    }

    {
        DoFRenumbering::Cuthill_McKee(dof_handler);
        DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
        SparsityPattern sparsity_pattern;
        sparsity_pattern.copy_from(dynamic_sparsity_pattern);
        std::ofstream out("sparsity_pattern2.svg");
        sparsity_pattern.print_svg(out);
    }
     */
}


void SolverParallel::assemble_system(function_sigma get_sigma, function_j_ext get_j_ext)
{
    TimerOutput::Scope t(computing_timer, "assemble");

    QGauss<3>   quadrature_formula(quad_n_points);
    QGauss<2> face_quadrature_formula(quad_n_points);
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int dofs_per_face = fe.dofs_per_face;
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs1(dofs_per_cell);
    Vector<double>       cell_rhs2(dofs_per_cell);
    std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
    std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FEValues<3>  fe_values (fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<3> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_gradients |
                                      update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);

    // debug bc
    /*
    std::set<types::global_dof_index> all_boundary_face_dofs;
    for (auto cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
    {
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }
        fe_values.reinit(cell);
        cell->get_dof_indices(cell_dof_indices);
        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (face->at_boundary())
            {
                fe_face_values.reinit(cell, face_number);

                face->get_dof_indices(face_dof_indices);

                for (auto i: face_dof_indices) {
                    std::cout << "at boundary " << i << std::endl;
                    all_boundary_face_dofs.insert(i);
                }
            }
        }
    }
    */
    // end debug

    typename DoFHandler<3>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        // process only locally owned cells in parallel version
        // if (not cell->is_locally_owned())
        // {
        //    continue;
        // }
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }

        cell_matrix = 0;
        cell_rhs1 = 0;
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            auto qp = fe_values.quadrature_point(q_point);
            double sigma = get_sigma(qp);
            auto JxW_at_qp = fe_values.JxW(q_point);
            auto j_ext = get_j_ext(qp);
            if (std::isnan(j_ext[0]) or std::isnan(j_ext[1]) or std::isnan(j_ext[2]))
            {
                std::cout << "NAN" << qp << "\t" << j_ext << std::endl;
                exit(-1);
            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    // Вроде как можно не учитывать некоторые граничные элементы, т.к.
                    // их начения известны из граничных условий Дирихле,
                    // и по этой причине в матрице системы все равно будет только один
                    // диагональный ненулевой элемент.
                    // Но экономия небольшая, а общность подхода уменьшается,
                    // поэтому это использоваться не будет.
                    // if (all_boundary_face_dofs.find(local_dof_indices[i]) != all_boundary_face_dofs.end() and
                    //    all_boundary_face_dofs.find(local_dof_indices[j]) != all_boundary_face_dofs.end())
                    // {
                    //     continue;
                    // }

                    cell_matrix(i, j) += sigma *
                                         ((fe_values.shape_grad(i, q_point) *
                                           fe_values.shape_grad(j, q_point)) *
                                          // fe_values.JxW(q_point));
                                          JxW_at_qp);
                }
                cell_rhs1(i) += (fe_values.shape_grad(i, q_point) *
                                j_ext *
                                // fe_values.JxW(q_point));
                                JxW_at_qp);

                // do nothing for cell_rhs2
            }
        }

        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            if (cell->face(face_number)->at_boundary() and
                cell->face(face_number)->boundary_id() == outer_boundary_id)
            {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                {
                    auto qp = fe_face_values.quadrature_point(q_point);
                    double sigma = get_sigma(qp);
                    auto JxW_at_qp = fe_face_values.JxW(q_point);
                    // auto j_ext = get_j_ext(qp);
                    auto normal = fe_face_values.normal_vector(q_point);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            //if (all_boundary_face_dofs.find(local_dof_indices[i]) != all_boundary_face_dofs.end() and
                            //    all_boundary_face_dofs.find(local_dof_indices[j]) != all_boundary_face_dofs.end())
                            // {
                            //     continue;
                            // }

                            cell_matrix(i, j) -= sigma *
                                                 fe_face_values.shape_value(i, q_point) *
                                                 (fe_face_values.shape_grad(j, q_point) * normal) *
                                                 JxW_at_qp;
                        }
                    }
                }
            }
        }

        constraints_j_ext.distribute_local_to_global(cell_matrix,
                                                     cell_rhs1,
                                                     local_dof_indices,
                                                     system_j_ext_matrix,
                                                     system_j_ext_rhs);
        cell_rhs2 = 0;
        constraints_b_0_1.distribute_local_to_global(cell_matrix,
                                                     cell_rhs2,
                                                     local_dof_indices,
                                                     system_b_0_1_matrix,
                                                     system_b_0_1_rhs);
    }

    system_j_ext_matrix.compress(VectorOperation::add);
    system_b_0_1_matrix.compress(VectorOperation::add);
    system_j_ext_rhs.compress(VectorOperation::add);
    system_b_0_1_rhs.compress(VectorOperation::add);
}


void SolverParallel::solve_system()
{

    if (do_save_matrix)
    {
        const std::string filename =
                ("system_j_ext_matrix." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream mof(filename.c_str());
        mof.precision(15);
        system_j_ext_matrix.print(mof);
        mof.close();
    }
    if (do_save_matrix)
    {
        const std::string filename =
                ("system_b_0_1_matrix." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream mof(filename.c_str());
        mof.precision(15);
        system_b_0_1_matrix.print(mof);
        mof.close();
    }
    if (do_save_rhs)
    {
        const std::string filename =
                ("system_j_ext_rhs." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream rhf(filename.c_str());
        system_j_ext_rhs.print(rhf, 15);
        rhf.close();
    }

    if (do_save_rhs)
    {
        const std::string filename =
                ("system_b_0_1_rhs." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream rhf(filename.c_str());
        system_b_0_1_rhs.print(rhf, 15);
        rhf.close();
    }

    {
        TimerOutput::Scope t(computing_timer, "solve_j_ext");

        SolverControl solver_control(dof_handler.n_dofs(), 1e-12, true, true);

        #ifdef USE_PETSC_LA
            LA::SolverCG solver(solver_control, mpi_communicator);
        #else
            LA::SolverCG solver(solver_control);
        #endif

            PETScWrappers::PreconditionEisenstat preconditioner;
            preconditioner.initialize(system_j_ext_matrix);

        #ifdef USE_PETSC_LA
            // data.symmetric_operator = true;
        #else
            // Trilinos defaults are good
        #endif

        LA::MPI::Vector
        completely_distributed_solution(locally_owned_dofs, mpi_communicator);

        solver.solve(system_j_ext_matrix, completely_distributed_solution, system_j_ext_rhs,
                     preconditioner);

        pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

        constraints_j_ext.distribute(completely_distributed_solution);

        locally_relevant_solution_j_ext = completely_distributed_solution;
    }

    {
        TimerOutput::Scope t(computing_timer, "solve_b_0_1");

        SolverControl solver_control(dof_handler.n_dofs(), 1e-12, true, true);

        #ifdef USE_PETSC_LA
            LA::SolverCG solver(solver_control, mpi_communicator);
        #else
            LA::SolverCG solver(solver_control);
        #endif

            PETScWrappers::PreconditionEisenstat preconditioner;
            preconditioner.initialize(system_b_0_1_matrix);

        #ifdef USE_PETSC_LA
            // data.symmetric_operator = true;
        #else
            // Trilinos defaults are good
        #endif

        LA::MPI::Vector
        completely_distributed_solution(locally_owned_dofs, mpi_communicator);

        // here we can calculate an initial guess for completely_distributed_solution
        // r0 = 6371000;
        // r1 = 6441000;
        // sigma0 = 5e-14;
        // h = 5000;
        // C3 = sigma0/(h*(exp((r0 - r1)/h) - 1));
        // C4 = -1/(exp((r0 - r1)/h) - 1);
        // rr_ex = r0:100:r1;
        // phi_ex = C4 + (C3*h*exp(-(rr_ex - r0)/h))/sigma0;
        {
            // std::cout
            //  << "Precalculation of initial guess for phi_b_0_1... "<< now_str()
            //  << std::endl;
            auto phi_b_0_1_estimation = [=](const dealii::Point<3> &p) -> double
            {
                constexpr double sigma0 = 5e-14;
                constexpr double h = 5000;
                const double C3 = sigma0/(h*(exp((r0 - r1)/h) - 1));
                const double C4 = -1/(exp((r0 - r1)/h) - 1);
                return C4 + (C3*h*exp(-(polar::r(p) - r0)/h))/sigma0;
            };
            interpolate_function(phi_b_0_1_estimation, completely_distributed_solution);
            // std::cout << "Precalculated initial guess for phi_b_0_1 " << now_str() << std::endl;
        }
        /*{
            LA::MPI::Vector locally_relevant_sol;
            locally_relevant_sol.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            locally_relevant_sol = completely_distributed_solution;
            
            const std::string filename =
                    ("sol_before_b_0_1_phi." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            locally_relevant_sol.print(rhf, 15);
            rhf.close();
        }*/

        solver.solve(system_b_0_1_matrix, completely_distributed_solution, system_b_0_1_rhs,
                     preconditioner);

        pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

        constraints_b_0_1.distribute(completely_distributed_solution);

        /*{
            LA::MPI::Vector locally_relevant_sol;
            locally_relevant_sol.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            locally_relevant_sol = completely_distributed_solution;

            const std::string filename =
                    ("sol_after_b_0_1_phi." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            locally_relevant_sol.print(rhf, 15);
            rhf.close();
        }*/

        locally_relevant_solution_b_0_1 = completely_distributed_solution;
    }
}


void SolverParallel::save_solution_vector(const LA::MPI::Vector &v,
                                          const char *name_prefix,
                                          const char *title)
{
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(v, title);

    data_out.build_patches();

    const std::string filename = (std::string(name_prefix) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
        {
          filenames.push_back(std::string(name_prefix) +
                              "." +
                              Utilities::int_to_string (i, 4) +
                              ".vtu");
        }

        std::ofstream master_output((std::string(name_prefix) + ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
    }

}

void SolverParallel::save_vector_field(const dealii::Vector<double> &v,
                                          const char *name_prefix,
                                          const char *title)
{

    dealii::FESystem<3, 3> gradients_fe(fe, 3);
    dealii::DoFHandler<3> gradient_dof_handler(dof_handler.get_triangulation());
    gradient_dof_handler.distribute_dofs(gradients_fe);
    DataOut<3> data_out;
    data_out.attach_dof_handler(gradient_dof_handler);

    std::vector<std::string> solution_names;
    solution_names.emplace_back(std::string(title) + "_x");
    solution_names.emplace_back(std::string(title) + "_y");
    solution_names.emplace_back(std::string(title) + "_z");

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(v, solution_names,
                             dealii::DataOut<3>::type_dof_data, data_component_interpretation);
    data_out.build_patches();

    const std::string filename = (std::string(name_prefix) +
                                  "." +
                                  Utilities::int_to_string
                                          (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
        {
            filenames.push_back(std::string(name_prefix) +
                                "." +
                                Utilities::int_to_string(i, 4) +
                                ".vtu");
        }

        std::ofstream master_output((std::string(name_prefix) + ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
    }

    std::cout << "saving done" << std::endl;
}


void SolverParallel::save_solution_j_ext(const char *name_prefix, const char *title)
{
    save_solution_vector(locally_relevant_solution_j_ext, name_prefix, title);
}

void SolverParallel::save_solution_b_0_1(const char *name_prefix, const char *title)
{
    save_solution_vector(locally_relevant_solution_b_0_1, name_prefix, title);
}

void SolverParallel::save_solution(const char *name_prefix, const char *title)
{
    save_solution_vector(locally_relevant_solution, name_prefix, title);
}


double SolverParallel::calc_boundary_integral_j_ext(dealii::types::boundary_id bid)
{
    TimerOutput::Scope t(computing_timer, "calc_boundary_integral_j_ext");
    return calc_boundary_integral(bid, locally_relevant_solution_j_ext);
}


double SolverParallel::calc_boundary_integral_b_0_1(dealii::types::boundary_id bid)
{
    TimerOutput::Scope t(computing_timer, "calc_boundary_integral_b_0_1");
    return calc_boundary_integral(bid, locally_relevant_solution_b_0_1);
}


double SolverParallel::calc_boundary_integral(dealii::types::boundary_id bid,
                                              const LA::MPI::Vector &solution)
{
    constexpr bool log_calc = false;

    pcout << "Calculate boundary integral..."
          << std::endl;

    // QSimpson<2> face_quadrature_formula;
    QGauss<2> face_quadrature_formula(quad_n_points);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_face = fe.dofs_per_face;
    // const unsigned int dofs_per_cell = fe.dofs_per_cell;

    std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
    // std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);

    LogFile calc_log(std::string("boundary_integral_calc_log_") +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) + ".txt",
                     log_calc);

    calc_log << "before create fe_face_values" << std::endl;

    FEFaceValues<3> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_gradients | update_quadrature_points |
                                   update_normal_vectors | update_JxW_values);

    calc_log << "Before loop over cell" << std::endl;

    double r = 0.0;

    typename DoFHandler<3>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    int cell_index = 0;
    for (; cell!=endc; ++cell, ++cell_index)
    {
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }


        calc_log << "local_cell " << cell_index << std::endl;

        // cell->get_dof_indices(cell_dof_indices);

        /*
        // check if face on boundary
        bool face_near_boundary = false;
        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (face->at_boundary() && (face->boundary_id() == bid))
            {
                face_near_boundary = true;
                break;
            }
        }

        if (not face_near_boundary)
        {
            continue;
        }

        fe_values.reinit(cell);

        // calculate gradient on cell in q_points
        for (unsigned int q_point=0; q_point<n_cell_q_points; ++q_point)
        {
            auto qp = fe_values.quadrature_point(q_point);

            Tensor<1, 3> gradient;
            for (unsigned int i=0; i<dofs_per_face; ++i)
            {
                calc_log << "\t" << cell_dof_indices[i] << "\t"
                         << solution[cell_dof_indices[i]] << "\t" << fe_values.shape_grad(i, q_point) << std::endl;
                gradient += fe_values.shape_grad(i, q_point) * solution[cell_dof_indices[i]];
            }
            calc_log << "  gradient @ qp " << qp << "\t:\t" << gradient << std::endl;
        }
        */

        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (face->at_boundary() && (face->boundary_id() == bid))
            {
                double r_face = 0;
                fe_face_values.reinit(cell, face_number);

                face->get_dof_indices(face_dof_indices);

                std::vector<Tensor<1, 3>> grads(face_quadrature_formula.size());

                fe_face_values.get_function_gradients(solution, grads);

                calc_log << "cell " << cell_index << "\tface " << face_number
                         << std::endl;

                if (log_calc)
                {
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        auto qp = fe_face_values.quadrature_point(q_point);
                        calc_log << "\t" << qp << "\t" << grads[q_point] << std::endl;
                    }
                }

                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    auto normal = fe_face_values.normal_vector(q_point);
                    auto qp = fe_face_values.quadrature_point(q_point);
                    double sigma = get_sigma(qp);

                    /*
                    Tensor<1, 3> gradient;

                    calc_log << "calculating gradient @ qp" << qp << "\t" << gradient << std::endl;

                    for (unsigned int i=0; i<dofs_per_face; ++i)
                    {
                        calc_log << "\t" << face_dof_indices[i] << "\t"
                                 << solution[face_dof_indices[i]] << std::endl;
                        gradient += fe_face_values.shape_grad(i, q_point) * solution[face_dof_indices[i]];
                    }

                    calc_log << "calculating face integral " << std::endl;
                    calc_log << sigma << "\t" << gradient << "\t"
                             << fe_face_values.JxW(q_point) << normal << std::endl;
                    */

                    r_face += sigma * grads[q_point] * fe_face_values.JxW(q_point) * normal;

                    calc_log << "face integral " << r_face << std::endl;
                }

                r += r_face;

                calc_log << "current result " << r << std::endl;
            }
        }
    }

    double rc = Utilities::MPI::sum(r, mpi_communicator);

    calc_log << "common result " << rc << std::endl;

    pcout << "... boundary integral calculated"
          << rc
          << std::endl;

    return rc;
}


double SolverParallel::calc_full_current(dealii::types::boundary_id bid,
                                         double sign,
                                         const LA::MPI::Vector &solution)
{
    constexpr bool log_calc = false;

    pcout << "Calculate full current..."
          << std::endl;

    QGauss<2> face_quadrature_formula(quad_n_points);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_face = fe.dofs_per_face;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
    std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);

    LogFile calc_log(std::string("full_current_calc_log_") +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) + ".txt",
                     log_calc);

    calc_log << "before create fe_face_values" << std::endl;

    FEFaceValues<3> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_gradients | update_quadrature_points |
                                   update_normal_vectors | update_JxW_values);

    calc_log << "Before loop over cell" << std::endl;

    double r = 0.0;

    typename DoFHandler<3>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    int cell_index = 0;
    for (; cell!=endc; ++cell, ++cell_index)
    {
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }

        calc_log << "local_cell " << cell_index << std::endl;

        cell->get_dof_indices(cell_dof_indices);

        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (face->at_boundary() && (face->boundary_id() == bid))
            {
                double r_face = 0;
                fe_face_values.reinit(cell, face_number);

                face->get_dof_indices(face_dof_indices);

                std::vector<Tensor<1, 3>> grads(face_quadrature_formula.size());

                fe_face_values.get_function_gradients(solution, grads);

                calc_log << "cell " << cell_index << "\tface " << face_number
                         << std::endl;

                if (log_calc)
                {
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        auto qp = fe_face_values.quadrature_point(q_point);
                            calc_log << "\t" << qp << "\t" << grads[q_point] << std::endl;
                    }
                }

                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    auto normal = fe_face_values.normal_vector(q_point);
                    auto qp = fe_face_values.quadrature_point(q_point);
                    double sigma = get_sigma(qp);

                    auto t = sigma * grads[q_point] * fe_face_values.JxW(q_point) * normal;
                    if (sign*t>0)
                    {
                        calc_log << "current positive " << qp << " " << t << std::endl;
                        r_face += t;
                    }
                    else if (sign*t<0)
                    {
                        calc_log << "current negative " << qp << " " << t << std::endl;
                    }
                    else
                    {
                        calc_log << "current zero " << t << std::endl;
                    }

                    calc_log << "face integral " << qp << " " << r_face << std::endl;
                }

                r += r_face;

                calc_log << "current result " << r << std::endl;
            }
        }
    }

    double rc = Utilities::MPI::sum(r, mpi_communicator);

    calc_log << "common result " << rc << std::endl;

    pcout << "... full current calculated"
          << rc
          << std::endl;

    return rc;
}


void SolverParallel::interpolate_function(std::function<double(const dealii::Point<3> &)> f,
                                          LA::MPI::Vector &completely_distributed_conductivity)
{

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    QTrapez<3>   quadrature_formula;
    FEValues<3>  fe_values(fe, quadrature_formula,
                           update_quadrature_points);

    typename DoFHandler<3>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();

    for (; cell != endc; ++cell)
    {
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        const std::vector<Point<3>> &points = fe_values.get_quadrature_points();

        for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
        {
            completely_distributed_conductivity[local_dof_indices[i]] = f(points[i]);
        }
    }

    completely_distributed_conductivity.compress(VectorOperation::insert);
}


void SolverParallel::interpolate_function(std::function<double(const dealii::Point<3> &)> f,
                                          const char *name_prefix, const char *title)
{
    pcout << "Calc " << title << "..."
          << std::endl;

    LA::MPI::Vector
        completely_distributed_conductivity(locally_owned_dofs, mpi_communicator);

    interpolate_function(std::move(f), completely_distributed_conductivity);

    LA::MPI::Vector             locally_relevant_conductivity;

    locally_relevant_conductivity.reinit(locally_owned_dofs,
                                         locally_relevant_dofs, mpi_communicator);

    locally_relevant_conductivity = completely_distributed_conductivity;

    save_solution_vector(locally_relevant_conductivity, name_prefix, title);

    pcout << "... " << title << " calculated"
          << std::endl;
}


void SolverParallel::calc_solution(double ionosphere_potential)
{
    constexpr bool do_save_solution = false;
    if (do_save_solution)
    {
        const std::string filename =
                ("sol_j_ext." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream rhf(filename.c_str());
        locally_relevant_solution_j_ext.print(rhf, 15);
        rhf.close();
    }
    if (do_save_solution)
    {
        const std::string filename =
                ("sol_b_0_1." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream rhf(filename.c_str());
        locally_relevant_solution_b_0_1.print(rhf, 15);
        rhf.close();
    }

    {
        dealii::Vector<double> sol1;
        sol1 = locally_relevant_solution_j_ext;

        dealii::Vector<double> sol2;
        sol2 = locally_relevant_solution_b_0_1;

        if (do_save_solution)
        {
            const std::string filename =
                    ("sol1_result_serial." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            sol1.print(rhf, 15);
            rhf.close();
        }
        if (do_save_solution)
        {
            const std::string filename =
                    ("sol2_result_serial." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            sol2.print(rhf, 15);
            rhf.close();
        }

        dealii::Vector<double> sol;
        sol=sol1;
        sol2 *= ionosphere_potential;

        if (do_save_solution)
        {
            const std::string filename =
                    ("sol2_scaled_result_serial." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            sol2.print(rhf, 15);
            rhf.close();
        }

        sol += sol2;

        if (do_save_solution)
        {
            const std::string filename =
                    ("sol_result_serial." +
                     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                     ".txt");
            std::ofstream rhf(filename.c_str());
            sol.print(rhf, 15);
            rhf.close();
        }
    }

    LA::MPI::Vector
        d_sol(locally_owned_dofs, mpi_communicator);
    LA::MPI::Vector
        d_sol_b_0_1(locally_owned_dofs, mpi_communicator);

    d_sol = locally_relevant_solution_j_ext;
    d_sol_b_0_1 = locally_relevant_solution_b_0_1;

    d_sol.add(ionosphere_potential, d_sol_b_0_1);

    locally_relevant_solution = d_sol;

    if (do_save_solution)
    {
        const std::string filename =
                ("sol_result." +
                 Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
                 ".txt");
        std::ofstream rhf(filename.c_str());
        locally_relevant_solution.print(rhf, 15);
        rhf.close();
    }
}

void SolverParallel::calc_save_electric_field(const LA::MPI::Vector &solution,
                                              const char *name_title)
{
    std::cout << "start calculating xyz gradients ... " << now_str() << std::endl;

    QTrapez<3> quadrature_formula;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    FEValues<3>  fe_values(fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

    dealii::Vector<double> gradients(solution.size()*3);
    std::vector<int> n_gradients(solution.size());
    std::vector<double> solution_values(solution.size());
    std::vector<int> point_touched(solution.size());
    std::vector<dealii::Point<3>> fe_points(solution.size());
    
    for (auto cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
    {
        if (not cell->is_locally_owned())
        {
            continue;
        }

        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        const std::vector<Point<3>> &points = fe_values.get_quadrature_points();

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            auto ii=dof_indices[q_point];
            const dealii::Point<3> &p = points[q_point];

            for (unsigned int i=0; i<dofs_per_cell; i++)
            {
                auto sg = fe_values.shape_grad(i, q_point);
                auto si = solution[dof_indices[i]];
                //gradients_value[ii] +=sg * si;
                gradients[3*ii+0] += sg[0] * si;
                gradients[3*ii+1] += sg[1] * si;
                gradients[3*ii+2] += sg[2] * si;
            }

            fe_points[ii] = p;
            ++n_gradients[ii];
            point_touched[ii] = 1;
            solution_values[ii] = solution[ii];
        }
    }

    /*{
        std::string fn = std::string(name_title)+"_"+to_string(get_mpi_pid())+ ".txt";
        std::ofstream f(fn.c_str());
        f.precision(8);
        for (unsigned i = 0; i < solution.size(); ++i)
        {
            f << i << "\t"
              << fe_points[i][0] << "\t"
              << fe_points[i][1] << "\t"
              << fe_points[i][2] << "\t"
              << gradients[i*3+0] << "\t"
              << gradients[i*3+1] << "\t"
              << gradients[i*3+2] << "\t"
              << n_gradients[i]
              << std::endl;
        }
        f.close();
    }
    */
    {
        std::string fn = std::string(name_title)+"_"+to_string(get_mpi_pid())+ ".dat";
        std::ofstream f(fn.c_str(), std::ios::binary);
        for (unsigned i = 0; i < solution.size(); ++i)
        {
            f.write(reinterpret_cast<char*>(&fe_points[i][0]), sizeof(double));
            f.write(reinterpret_cast<char*>(&fe_points[i][1]), sizeof(double));
            f.write(reinterpret_cast<char*>(&fe_points[i][2]), sizeof(double));
            f.write(reinterpret_cast<char*>(&solution_values[i]), sizeof(double));
            f.write(reinterpret_cast<char*>(&gradients[i*3+0]), sizeof(double));
            f.write(reinterpret_cast<char*>(&gradients[i*3+1]), sizeof(double));
            f.write(reinterpret_cast<char*>(&gradients[i*3+2]), sizeof(double));
            double n_gradients_current = n_gradients[i];
            f.write(reinterpret_cast<char*>(&n_gradients_current), sizeof(double));
        }
        f.close();
    }

    for (unsigned i = 0; i < solution.size(); ++i)
    {
        //gradients_value[i] /=n_gradients[i];
        gradients[i*3+0] /= n_gradients[i];
        gradients[i*3+1] /= n_gradients[i];
        gradients[i*3+2] /= n_gradients[i];
    }

    std::cout << "... end of calculating gradients, saving ..." << now_str() << std::endl;
    save_vector_field(gradients, name_title, "E");
    std::cout << "... gradients saved " << now_str() << std::endl;
}

void SolverParallel::calc_save_electric_field_enz(const LA::MPI::Vector &solution,
                                                  const char *name_title)
{
    std::cout << "start calculating enz gradients... " << now_str() << std::endl;

    QTrapez<3> quadrature_formula;
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    // std::vector<Tensor<1, 3>> gradients_value;
    // gradients_value.reserve(solution.size());

    dealii::Vector<double> gradients;
    gradients.reinit(solution.size()*3);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    FEValues<3>  fe_values(fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
    std::vector<int> n_gradients(solution.size());
    std::vector<dealii::Point<3>> fe_points(solution.size());

    for (auto cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
    {
        if (not cell->is_locally_owned())
        {
            continue;
        }

        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        const std::vector<Point<3>> &points = fe_values.get_quadrature_points();

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            auto ii=dof_indices[q_point];
            const dealii::Point<3> &p = points[q_point];

            for (unsigned int i=0; i<dofs_per_cell; i++)
            {
                auto sg = fe_values.shape_grad(i, q_point);
                auto si = solution[dof_indices[i]];
                gradients[3*ii+0] += sg[0]*si;
                gradients[3*ii+1] += sg[1]*si;
                gradients[3*ii+2] += sg[2]*si;
            }

            fe_points[ii] = p;
            ++n_gradients[ii];
        }
    }

    for (unsigned i = 0; i < solution.size(); ++i)
    {
        gradients[i*3+0] /= n_gradients[i];
        gradients[i*3+1] /= n_gradients[i];
        gradients[i*3+2] /= n_gradients[i];

        double gx = gradients[i*3+0];
        double gy = gradients[i*3+1];
        double gz = gradients[i*3+2];

        double norm_before_roll = sqrt(sqr(gx)+sqr(gy)+sqr(gz));

        dealii::Tensor<1, 3> enz_gradient = xyz_to_enz(fe_points[i],
                                                       dealii::Tensor<1, 3>({gx, gy, gz}));
        gradients[i*3+0] = enz_gradient[0];
        gradients[i*3+1] = enz_gradient[1];
        gradients[i*3+2] = enz_gradient[2];
        double norm_after_roll = sqrt(sqr(gradients[i*3])+sqr(gradients[i*3+1])+sqr(gradients[i*3+2]));

        double eps = std::max(std::max(norm_before_roll, norm_after_roll), 1.0);
        if (not (near(norm_before_roll, norm_after_roll, eps*1e-10)))
        {
            std::cout << "wrong gradient roll "
                    << "@ " << fe_points[i] << " "
                    << "("
                    << gx << " "
                    << gy << " "
                    << gz << ") "
                    << "|" << norm_before_roll << "|"
                    << " != "
                    << "("
                    << gradients[i*3] << " "
                    << gradients[i*3+1] << " "
                    << gradients[i*3+2] << ") "
                    << "|" << norm_after_roll << "|"
                    << " eps = " << eps
                    << std::endl;
        }
    }

    {
        std::string fn = std::string("E_enz_"+to_string(get_mpi_pid())+ ".txt");
        std::ofstream f(fn.c_str());
        f.precision(15);
        for (unsigned i = 0; i < solution.size(); ++i)
        {
            const dealii::Point<3, double> &p = fe_points[i];
            double r = polar::r(p);
            double theta = polar::theta(p);
            double phi = polar::phi(p);

            f << i << "\t"
                    << p[0] << "\t"
                    << p[1] << "\t"
                    << p[2] << "\t"
                    << r << "\t" << theta << "\t" << phi << "\t"
              << gradients[i*3] << "\t"
              << gradients[i*3+1] << "\t"
              << gradients[i*3+2] << "\t"
              << n_gradients[i]
              << std::endl;
        }
        f.close();
    }

    std::cout << "... end of calculating gradients, saving ... " << now_str() << std::endl;
    save_vector_field(gradients, name_title, "E_enz");
    std::cout << "... gradients saved " << now_str() << std::endl;
}


double SolverParallel::test_integral_over_cells_boundaries(const LA::MPI::Vector &solution,
                                                           function_j_ext j_ext)
{
    TimerOutput::Scope t(computing_timer, "test_integral_over_cells_boundaries");

    QGauss<2> face_quadrature_formula(quad_n_points);

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FEFaceValues<3> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);

    double r_face_max = 0;

    typename DoFHandler<3>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    int cell_index = 0;
    for (; cell!=endc; ++cell, ++cell_index)
    {
        if (cell->subdomain_id() != this_mpi_process)
        {
            continue;
        }

        double V = cell->measure();

        // @todo
        // div(sigma*grad(phi) - j_ext) = 0 <=> Phi/V = 0
        // so Phi/V = 1/V(Cell) *
        //              * integral by CellBoundary of ((sigma*grad(phi) - j_ext), n) by ds = 0

        double r_face = 0;

        // loop by faces of current cell
        for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
        {
            fe_face_values.reinit(cell, face_number);

            std::vector<Tensor<1, 3>> grads(face_quadrature_formula.size());
            fe_face_values.get_function_gradients(solution, grads);

            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            {
                auto normal = fe_face_values.normal_vector(q_point);
                auto qp = fe_face_values.quadrature_point(q_point);
                auto jp = j_ext(qp);
                double sigma = get_sigma(qp);
                r_face += (sigma * grads[q_point] -jp)* fe_face_values.JxW(q_point) * normal;
            }

        }
        r_face /= V;

        if (fabs(r_face) > r_face_max)
        {
            r_face_max = r_face;
        }

        {
            std::string fn = std::string("boundary_flow_") + to_string(get_mpi_pid()) + ".txt";
            std::ofstream of(fn.c_str(), std::ios::app);
            auto c = cell->center();
            of << cell_index << "\t"
                << c << "\t"
                << polar::r(c) << "\t" << polar::lambda(c) << "\t" << polar::phi(c) << "\t"
                << r_face << std::endl;
            of.close();
        }
    }

    return r_face_max;
}

double SolverParallel::test_kelly(const LA::MPI::Vector &solution, function_j_ext j_ext)
{
    (void)j_ext;
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<3>::estimate(dof_handler,
                                       QGauss<2>(3),
                                       typename FunctionMap<3>::type(),
                                       solution,
                                       estimated_error_per_cell);
    double s(0.0);
    for (float e : estimated_error_per_cell)
    {
        s += e;
    }
    return s;
}


void SolverParallel::save_potential_on_slice(const LA::MPI::Vector &solution)
{
    auto &os = std::cout;

    Functions::FEFieldFunction<3, DoFHandler<3>, LA::MPI::Vector> ff(dof_handler, solution);

    size_t N = 10;
    double lat_from = 0;
    double lat_to = 90;
    double dlat = (lat_to-lat_from) / (N-1);
    std::vector<double> lat(N);
    for (size_t i=0; i<N; ++i) { lat[i] = lat_from+i*dlat; }
    std::vector<dealii::Tensor<1, 3>> gradients(N); // from 0 to 90 degrees

    os.precision(17);

    for (double z = r0; z<r1; z+=100)
    {
        for (size_t i = 0; i < N; ++i)
        {
            Point<3> p(z * cos(lat[i]), 0, z * sin(lat[i]));
            // dealii::Tensor<1, 3> g;
            double v;
            try
            {
                // g = ff.gradient(p);
                v = ff.value(p);
            }
            catch (const VectorTools::ExcPointNotAvailableHere &)
            {
                // g = 0;
                v = 0;
                os << "phi slice " << get_mpi_pid() << "\t"
                   << z << "\t" << lat[i] << "\t"
                   << "VectorTools::ExcPointNotAvailableHere" << std::endl;
            }
            catch (const std::exception & ex)
            {
                // g = 0;
                v = 0;
                os << "phi slice " << get_mpi_pid() << "\t"
                   << z << "\t" << lat[i] << "\t" << "std::exception " << ex.what() << std::endl;
            }
            catch (...)
            {
                v = 0;
                os << "phi slice " << get_mpi_pid() << "\t"
                   << z << "\t" << lat[i] << "\t" << "some other exception" << std::endl;
            }

            // double vs = Utilities::MPI::sum(v, mpi_communicator);

            os << "phi slice " << get_mpi_pid() << "\t"
               << z << "\t" << lat[i] << "\t" << v << std::endl;
        }
    }
}
