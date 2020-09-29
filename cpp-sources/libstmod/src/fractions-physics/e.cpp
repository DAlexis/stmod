#include "stmod/fractions-physics/e.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/matgen.hpp"
#include "stmod/grid/grid.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_q.h>

using namespace dealii;

const std::string Electrons::m_names[2] = {"Electrons_density", "Electrons_density_derivative"};

Electrons::Electrons(const FEGlobalResources& fe_global_res) :
    Variable(m_names[0]), Fraction(m_names[0]), m_fe_global_res(fe_global_res)
{
}

void Electrons::set_electric_field(const dealii::Vector<double>& Ex, const dealii::Vector<double>& Ey, const dealii::Vector<double>& total_charge)
{
    m_Ex = &Ex;
    m_Ey = &Ey;
    m_total_charge = &total_charge;
}

const dealii::Vector<double>& Electrons::get_implicit_delta(double dt, double theta)
{
    // Creating implicit system matrix
    m_implicit_system_matrix.copy_from(m_fe_global_res.mass_matrix());

    m_tmp_matrix = 0;

    create_E_psi_grad_psi_matrix_axial(
            //10/0.002, 10/0.002,
            *m_Ex,
            *m_Ey,
            m_fe_global_res.dof_handler(),
            m_tmp_matrix);

    m_implicit_system_matrix.add(-dt*theta*parameters.D_e, m_fe_global_res.laplace_matrix());
    m_implicit_system_matrix.add(-dt*theta*parameters.mu_e * (-1.0), m_tmp_matrix);


    //m_implicit_system_matrix.add(-dt*theta*parameters.mu_e, m_E_grad_psi_psi_matrix);


    // Creating implicit RHS
    m_implicit_rhs_matrix = 0;
    m_implicit_rhs_matrix.add(dt*parameters.D_e, m_fe_global_res.laplace_matrix());
    m_implicit_rhs_matrix.add(dt*parameters.mu_e * (-1.0), m_tmp_matrix);
    //m_implicit_rhs_matrix.add(dt*parameters.mu_e, m_E_grad_psi_psi_matrix);

    m_implicit_rhs = 0;
    m_implicit_rhs_matrix.vmult(m_implicit_rhs, m_concentration);


    // Preparing to solve system
    m_implicit_delta = 0;
    m_fe_global_res.constraints().condense(m_implicit_system_matrix, m_implicit_rhs);

    // Applying boundary conditions
    MatrixTools::apply_boundary_values(m_boundary_values,
                                     m_implicit_system_matrix,
                                     m_implicit_delta,
                                     m_implicit_rhs);

    // Creating inverted matrix
    m_implicit_system_reversed.initialize(m_implicit_system_matrix);

    m_implicit_system_reversed.vmult(m_implicit_delta, m_implicit_rhs);
    m_fe_global_res.constraints().distribute(m_implicit_delta);
    return m_implicit_delta;
}

Fraction& Electrons::operator=(double value)
{
    Variable::operator =(value);
    m_fe_global_res.constraints().distribute(values_w());
    return *this;
}

void Electrons::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    Fraction::init_mesh_dependent(dof_handler);
    m_system_rhs.reinit(m_fe_global_res.n_dofs());

    m_implicit_delta.reinit(m_fe_global_res.n_dofs());
    m_implicit_rhs.reinit(m_fe_global_res.n_dofs());
    m_tmp_vector.reinit(m_fe_global_res.n_dofs());

    m_implicit_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_implicit_system_matrix.reinit(m_fe_global_res.sparsity_pattern());

    m_tmp_matrix.reinit(m_fe_global_res.sparsity_pattern());

    // Creating bounndary values map
    m_boundary_values.clear();
    // Boundary conditions for Delta is zero
    /*VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::top_and_needle,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);*/

    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::bottom,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);

    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::outer_border,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);
}

void Electrons::compute_derivatives(double t)
{
    Fraction::compute_derivatives(t);
/*    m_tmp = m_concentration; // Charge of electrons
    m_tmp.scale(*m_total_charge); // * total charge. Negative if only electrons in the system
    m_tmp *= -parameters.mu_e;
    m_derivative += m_tmp;
  */
}
