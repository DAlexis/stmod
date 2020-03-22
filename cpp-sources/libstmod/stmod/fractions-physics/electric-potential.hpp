#ifndef ELECTRIC_POTENTIAL_HPP_INCLUDED
#define ELECTRIC_POTENTIAL_HPP_INCLUDED

#include "stmod/i-output-provider.hpp"
#include "stmod/i-steppable.hpp"
#include "stmod/i-mesh-based.hpp"
#include "stmod/fe-common.hpp"

#include <deal.II/lac/sparse_direct.h>

struct ElectricParameters
{
    double needle_potential = 10.0;
    double bottom_potential = -3.0;
};

class ElectricPotential : public IOutputProvider, public IMeshBased, public IPreStepJob
{
public:
    ElectricPotential(const FEResources& fe_res);

    // IVariablesStorage
    virtual dealii::Vector<double>& values_vector() override;

    // IMeshBased
    void init_mesh_dependent() override;
    const dealii::Vector<double>& error_estimation_vector() const override;

    // IPreStepJob
    void compute(double t) override;

    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    virtual size_t output_values_count() const override;

    void add_charge(const dealii::Vector<double>& charge_vector, double mul = 1.0);

    const dealii::Vector<double>& total_chagre() const;

    void set_electric_parameters(const ElectricParameters& electric_parameters);

private:
    void calc_total_charge();

    /* @brief This function is boundary condition-dependent, so it should be called every potential change
     * It may be optimized later
     */
    void create_system_matrix_and_inverse_matrix();
    void create_rhs();
    void solve_lin_eq();

    void create_boundary_pattern();

    ElectricParameters m_electric_parameters;

    const FEResources& m_fe_res;

    dealii::AffineConstraints<double> m_potential_constraints;
    dealii::SparsityPattern  m_sparsity_pattern;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::SparseMatrix<double> m_mass_matrix;
    dealii::SparseDirectUMFPACK m_system_matrix_inverse;
    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_system_rhs_boundary;
    dealii::Vector<double> m_system_rhs_boundary_pattern; // 0.0 if is boundary value and 1.0 otherwise

    dealii::Vector<double> m_solution;

    std::vector<const dealii::Vector<double>*> m_charges;
    std::vector<double> m_charges_muls;
    dealii::Vector<double> m_total_charge;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    double m_needle_potential = 0, m_bottom_potential = 0;

    const std::string m_name = "Electric_potential";
};

#endif // ELECTRIC_POTENTIAL_HPP_INCLUDED
