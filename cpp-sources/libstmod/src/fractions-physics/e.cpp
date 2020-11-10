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

Electrons::CathodeSupressionFunction::CathodeSupressionFunction(double scale) :
    m_scale(scale)
{
}

double Electrons::CathodeSupressionFunction::value(const dealii::Point<2> &p, const unsigned int component) const
{
    if (p[1] > m_scale)
        return 1.0;

    return p[1] / m_scale;
}

Electrons::Electrons(const FEGlobalResources& fe_global_res) :
    ScalarVariable(m_names[0]), Fraction(m_names[0]), m_fe_global_res(fe_global_res)
{
}

Fraction& Electrons::operator=(double value)
{
    ScalarVariable::operator =(value);
    m_fe_global_res.constraints().distribute(values_w());
    return *this;
}

void Electrons::init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler)
{
    Fraction::init_mesh_dependent(dof_handler);
    m_system_rhs.reinit(m_fe_global_res.n_dofs());
    m_cathode_supression_field.reinit(m_fe_global_res.n_dofs());

    create_cathode_supression_field();

    // Creating bounndary values map
    m_boundary_values.clear();
    // Boundary conditions for Delta is zero

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
                                             m_boundary_values);
}

void Electrons::compute_derivatives(double t)
{
    Fraction::compute_derivatives(t);
    /*
    m_tmp = m_concentration;
    m_tmp.scale(*m_total_charge);
    m_tmp *= parameters.mu_e;
    m_derivative += m_tmp;*/
}

void Electrons::apply_cathode_supression()
{
    m_concentration.scale(m_cathode_supression_field);
}

void Electrons::create_cathode_supression_field()
{
    dealii::VectorTools::interpolate(m_fe_global_res.dof_handler(), CathodeSupressionFunction(0.0005), m_cathode_supression_field);
}
