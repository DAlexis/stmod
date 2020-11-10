#include "stmod/fractions-physics/divergence.hpp"
#include "stmod/matgen.hpp"
#include "stmod/grid/grid.hpp"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

Divergence::Divergence(
        const FEGlobalResources& fe_res,
        const dealii::Vector<double>& field_x,
        const dealii::Vector<double>& field_y,
        const std::string& name):
    ScalarVariable(name), m_fe_global_res(fe_res), m_field_x(field_x), m_field_y(field_y)
{
}

void Divergence::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    SecondaryValue::init_mesh_dependent(dof_handler);
    m_rhs.reinit(dof_handler.n_dofs());
    m_tmp.reinit(dof_handler.n_dofs());

    m_system_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_dfield_dx_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());
    m_dfield_dy_rhs_matrix.reinit(m_fe_global_res.sparsity_pattern());

    create_r_grad_phi_i_comp_phi_j_axial(m_fe_global_res.dof_handler(), m_dfield_dx_rhs_matrix, 0);
    create_r_grad_phi_i_comp_phi_j_axial(m_fe_global_res.dof_handler(), m_dfield_dy_rhs_matrix, 1);

    m_boundary_values.clear();
/*
    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::top_and_needle,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);

    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::bottom,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);

    VectorTools::interpolate_boundary_values(m_fe_global_res.dof_handler(),
                                             BoundaryAssigner::outer_border,
                                             Functions::ZeroFunction<2>(),
                                             m_boundary_values);*/
}

void Divergence::compute(double t)
{
    m_rhs = 0;
    m_tmp = 0;
    m_value = 0;

    m_dfield_dx_rhs_matrix.vmult(m_rhs, m_field_x);
    m_dfield_dy_rhs_matrix.vmult(m_tmp, m_field_y);
    m_rhs += m_tmp;

    m_system_matrix.copy_from(m_fe_global_res.mass_matrix());
    m_fe_global_res.constraints().condense(m_system_matrix, m_rhs);

    // Applying boundary conditions
    MatrixTools::apply_boundary_values(m_boundary_values,
                                     m_system_matrix,
                                     m_value,
                                     m_rhs);

    m_mass_matrix_inverse.initialize(m_system_matrix);
    m_mass_matrix_inverse.vmult(m_value, m_rhs);
    m_fe_global_res.constraints().distribute(m_value);
}
