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

struct ElectronsConstants
{
    double D_e = 0.1; //   m^2 c^-1      Diffusion coefficient
    double mu_e = 5.92; // m^2 V^-1 c^-1 Mobility
    double nu_i; // Ionization frequency
    double nu_a_1, nu_a_23; // Single-particle and double-particle attachment frequency
    double nu_d_u1, nu_d_u2, nu_d_s; // Electron detachment frequencies
};

struct ElectronsParameters
{
    double mu_e = -5.92; // m^2 V^-1 s^-1
    //double mu_e = 5.92e-3; // m^2 V^-1 s^-1
    //double mu_e = 0.0; // m^2 V^-1 s^-1
    double D_e = 0.1; // m^2 s^-1
    //double D_e = 1; // m^2 s^-1
    //double D_e = 0.0; // m^2 s^-1
};

class Electrons : public Fraction, public IImplicitSteppable
{
public:
    Electrons(const FEGlobalResources& fe_res);

    // IVariablesStorage
    dealii::Vector<double>& values_w() override;

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    const dealii::Vector<double>& derivative() const;

    dealii::Vector<double>& value_w();

    void set_potential(const dealii::Vector<double>& potential);

    const dealii::Vector<double>& get_implicit_delta(double dt, double theta = 0.5);

    Fraction& operator=(double value);

    ElectronsParameters parameters;

private:
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    const FEGlobalResources& m_fe_global_res;

    dealii::Vector<double> m_system_rhs;

    dealii::Vector<double> m_tmp_vector;
    dealii::Vector<double> m_implicit_rhs;

    dealii::SparseMatrix<double> m_E_grad_psi_psi_matrix;

    dealii::SparseMatrix<double> m_implicit_rhs_matrix;
    dealii::SparseMatrix<double> m_implicit_system_matrix;
    dealii::SparseMatrix<double> m_tmp_matrix;
    dealii::SparseDirectUMFPACK m_implicit_system_reversed;
    dealii::Vector<double> m_implicit_delta;

    const dealii::Vector<double>* m_potential = nullptr;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    const std::string m_names[2] = {"Electrons_density", "Electrons_density_derivative"};
};

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
