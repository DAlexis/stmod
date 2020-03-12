#ifndef FRACTIONS_PHYSICS_E_HPP_INCLUDED
#define FRACTIONS_PHYSICS_E_HPP_INCLUDED

#include "stmod/fractions.hpp"
#include "stmod/fractions-base.hpp"
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

class ElectronsRHS : public FractionBase
{
public:
    ElectronsRHS(
        FractionsStorage& storage,
        size_t fraction_index,
        const dealii::Vector<double>& potential_solution,
        const dealii::DoFHandler<2>& dof_handler
    );

    ElectronsConstants constants;

    void calculate_rhs(double time) override;

    int n_p_ind = -1;
    int n_n1_ind = -1;
    int n_n2_ind = -1;
    int n_n3_ind = -1;
    int n_n4_ind = -1;

private:
    const dealii::Vector<double>& m_potential;
    const dealii::DoFHandler<2>& m_dof_handler;

};

class ElectronIterator : public FractionBase
{
public:
    ElectronIterator(
        FractionsStorage& storage,
        size_t fraction_index,
        const dealii::Vector<double>& potential_solution,
        const dealii::DoFHandler<2>& dof_handler
    );

    ElectronsConstants constants;

private:
    dealii::SparseMatrix<double> m_laplace_matrix;
    dealii::SparseMatrix<double> m_mass_matrix;
    dealii::SparseMatrix<double> m_system_matrix;
    const dealii::DoFHandler<2>& m_dof_handler;
};

struct ElectronsParameters
{
    double mu_e = 5.92; // m^2 V^-1 s^-1
    double D_e = 0.1; // m^2 s^-1
};

class Electrons : public IFractionData
{
public:
    Electrons(const FEResources& fe_res);

    void init();
    void solve();

    const std::string& name(size_t index) const override;
    const dealii::Vector<double>& value(size_t index) const override;
    virtual size_t values_count() const override;
    const dealii::Vector<double>& derivative() const;

    dealii::Vector<double>& value_w();

    void add_single_source(double reaction_const, const dealii::Vector<double>& source);
    void add_pair_source(double reaction_const, const dealii::Vector<double>& source1, const dealii::Vector<double>& source2);
    void set_potential(const dealii::Vector<double>& potential);

    ElectronsParameters parameters;
private:
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    void assemble_system();
    void solve_lin_eq();

    void add_single_point_derivative();

    const FEResources& m_fe_res;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_concentration;
    dealii::Vector<double> m_derivative;
    dealii::Vector<double> m_derivative_without_single_point;

    dealii::Vector<double> m_tmp;

    std::vector<const dealii::Vector<double>*> m_single_sources;
    std::vector<double> m_single_reaction_consts;

    std::vector<PairSourceTuple> m_pair_sources;
    std::vector<double> m_pair_reaction_consts;

    const dealii::Vector<double>* m_potential = nullptr;

    const std::string m_names[2] = {"Electrons_density", "Electrons_density_derivative"};
};

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
