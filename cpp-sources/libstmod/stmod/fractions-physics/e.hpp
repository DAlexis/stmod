#ifndef FRACTIONS_PHYSICS_E_HPP_INCLUDED
#define FRACTIONS_PHYSICS_E_HPP_INCLUDED

#include "stmod/fractions.hpp"
#include "stmod/i-output-provider.hpp"
#include "stmod/i-steppable.hpp"
#include "stmod/i-mesh-based.hpp"

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

class Electrons : public IOutputProvider, public ISteppable, public IMeshBased
{
public:
    Electrons(const FEGlobalResources& fe_res);

    // IVariablesStorage
    dealii::Vector<double>& values_vector() override;

    // IMeshBased
    void init_mesh_dependent() override;
    const dealii::Vector<double>& error_estimation_vector() const override;

    // IOutputProvider
    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    virtual size_t output_values_count() const override;

    // ISteppable
    const dealii::Vector<double>& derivatives_vector() const override;
    void compute(double t) override;

    const dealii::Vector<double>& derivative() const;

    dealii::Vector<double>& value_w();

    void add_single_source(double reaction_const, const dealii::Vector<double>& source);
    void add_pair_source(double reaction_const, const dealii::Vector<double>& source1, const dealii::Vector<double>& source2);
    void set_potential_and_total_charge(const dealii::Vector<double>& potential, const dealii::Vector<double>& total_charge);

    const dealii::Vector<double>& get_implicit_dn(double dt, double theta = 0.5);

    ElectronsParameters parameters;

private:
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    void create_implicit_method_matrixes(double dt, double theta = 0.5);

    void create_rhs();
    void solve_lin_eq();

    void add_single_point_derivative();

    const FEGlobalResources& m_fe_global_res;

    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_concentration;
    dealii::Vector<double> m_derivative;
    dealii::Vector<double> m_derivative_without_single_point;

    dealii::Vector<double> m_tmp_vector;
    dealii::Vector<double> m_implicit_rhs;

    std::vector<const dealii::Vector<double>*> m_single_sources;
    std::vector<double> m_single_reaction_consts;

    std::vector<PairSourceTuple> m_pair_sources;
    std::vector<double> m_pair_reaction_consts;

    dealii::SparseMatrix<double> m_E_grad_psi_psi_matrix;

    dealii::SparseMatrix<double> m_implicit_rhs_matrix;
    dealii::SparseMatrix<double> m_implicit_system_matrix;
    dealii::SparseMatrix<double> m_tmp_matrix;
    dealii::SparseDirectUMFPACK m_implicit_system_reversed;
    dealii::Vector<double> m_implicit_delta;


    const dealii::Vector<double>* m_potential = nullptr;
    const dealii::Vector<double>* m_total_charge = nullptr;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    const std::string m_names[2] = {"Electrons_density", "Electrons_density_derivative"};
};

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
