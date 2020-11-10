#ifndef FRACTIONS_PHYSICS_E_HPP_INCLUDED
#define FRACTIONS_PHYSICS_E_HPP_INCLUDED

#include "stmod/fractions/fraction.hpp"
#include "stmod/output/output-provider.hpp"
#include "stmod/time/time-iterable.hpp"
#include "stmod/grid/mesh-based.hpp"

#include "stmod/fe-common.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

#include <vector>
#include <tuple>

struct ElectronsParameters
{
    double mu_e = 5.92; // m^2 V^-1 s^-1
    //double mu_e = 5.92e-3; // m^2 V^-1 s^-1
    //double mu_e = 0.0; // m^2 V^-1 s^-1
    //double D_e = 0.1; // m^2 s^-1
    double D_e = 0.1; // m^2 s^-1
    //double D_e = 1; // m^2 s^-1
    //double D_e = 0.0; // m^2 s^-1
};

class Electrons : public Fraction, public ImplicitSteppable
{
public:
    Electrons(const FEGlobalResources& fe_res);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    void compute_derivatives(double t) override;

    void set_electric_field(const dealii::Vector<double>& Ex, const dealii::Vector<double>& Ey, const dealii::Vector<double>& total_charge);

    void apply_boundary_to_concentration();

    const dealii::Vector<double>& get_implicit_delta(double dt, double theta = 0.5);

    Fraction& operator=(double value);

    ElectronsParameters parameters;

private:
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    const FEGlobalResources& m_fe_global_res;

    dealii::Vector<double> m_system_rhs;

    dealii::Vector<double> m_tmp_vector;
    dealii::Vector<double> m_implicit_rhs;

    dealii::SparseMatrix<double> m_implicit_rhs_matrix;
    dealii::SparseMatrix<double> m_implicit_system_matrix;
    dealii::SparseMatrix<double> m_tmp_matrix;
    dealii::SparseDirectUMFPACK m_implicit_system_reversed;
    dealii::Vector<double> m_implicit_delta;

    const dealii::Vector<double>* m_Ex = nullptr;
    const dealii::Vector<double>* m_Ey = nullptr;
    const dealii::Vector<double>* m_total_charge = nullptr;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    static const std::string m_names[2];
};

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
