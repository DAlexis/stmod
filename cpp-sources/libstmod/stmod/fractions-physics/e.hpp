#ifndef FRACTIONS_PHYSICS_E_HPP_INCLUDED
#define FRACTIONS_PHYSICS_E_HPP_INCLUDED

#include "stmod/fractions.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

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

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
