#ifndef ELECTRIC_POTENTIAL_HPP_INCLUDED
#define ELECTRIC_POTENTIAL_HPP_INCLUDED

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"
#include "stmod/fe-sampler.hpp"

#include <deal.II/lac/sparse_direct.h>

struct ElectricParameters
{
    double needle_potential = 10000.0;
    double bottom_potential = 0.0;
};

class ElectricPotential : public SecondaryValue
{
public:
    ElectricPotential(const FEGlobalResources& fe_res);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepComputer
    void compute(double t) override;

    void add_charge(const dealii::Vector<double>& charge_vector, double mul = 1.0);

    const dealii::Vector<double>& total_chagre() const;

    void set_electric_parameters(const ElectricParameters& electric_parameters);

    const std::vector<dealii::Tensor<1, 2>>& E_vector();
    const dealii::Vector<double>& E_scalar();
    const dealii::Vector<double>& E_x();
    const dealii::Vector<double>& E_y();

    const std::string& output_name(size_t index) const override;
    const dealii::Vector<double>& output_value(size_t index) const override;
    size_t output_values_count() const override;

private:
    void calc_total_charge();

    void create_e_field();
    void create_rhs_matrix(unsigned int component);

    ElectricParameters m_electric_parameters;

    const FEGlobalResources& m_fe_global_res;

    dealii::SparseMatrix<double> m_system_matrix;

    dealii::SparseMatrix<double> m_E_x_rhs_matrix;
    dealii::SparseMatrix<double> m_E_y_rhs_matrix;

    dealii::SparseDirectUMFPACK m_mass_matrix_inverse;

    dealii::SparseDirectUMFPACK m_system_matrix_inverse;

    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_E_x;
    dealii::Vector<double> m_E_y;

    dealii::Vector<double> m_Ex_rhs;
    dealii::Vector<double> m_Ey_rhs;

    dealii::Vector<double> m_E_scalar;

    FESampler m_electric_field_sampler;

    std::vector<const dealii::Vector<double>*> m_charges;
    std::vector<double> m_charges_muls;
    dealii::Vector<double> m_total_charge;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    double m_needle_potential = 0, m_bottom_potential = 0;

    static const std::string m_name;
    const std::string m_name_pot = "electric_field_phi";
    const std::string m_name_Ex = "electric_field_Ex";
    const std::string m_name_Ey = "electric_field_Ey";
    const std::string m_name_E = "electric_field_E";
};

#endif // ELECTRIC_POTENTIAL_HPP_INCLUDED
