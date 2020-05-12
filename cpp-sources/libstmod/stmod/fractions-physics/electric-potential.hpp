#ifndef ELECTRIC_POTENTIAL_HPP_INCLUDED
#define ELECTRIC_POTENTIAL_HPP_INCLUDED

#include "stmod/output/output-provider.hpp"
#include "stmod/time/time-iterable.hpp"
#include "stmod/grid/mesh-based.hpp"
#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

#include <deal.II/lac/sparse_direct.h>

struct ElectricParameters
{
    double needle_potential = 10.0;
    double bottom_potential = -3.0;
};

class ElectricPotential : public SecondaryValue//, public IPreStepJob
{
public:
    ElectricPotential(const FEGlobalResources& fe_res);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // // IPreStepJob
    void compute(double t) override;

    void add_charge(const dealii::Vector<double>& charge_vector, double mul = 1.0);

    const dealii::Vector<double>& total_chagre() const;

    void set_electric_parameters(const ElectricParameters& electric_parameters);

private:
    void calc_total_charge();

    /* @brief This function is boundary condition-dependent, so it should be called every potential change
     * It may be optimized later
     */
    void create_rhs();

    ElectricParameters m_electric_parameters;

    const FEGlobalResources& m_fe_global_res;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::SparseDirectUMFPACK m_system_matrix_inverse;
    dealii::Vector<double> m_system_rhs;

    std::vector<const dealii::Vector<double>*> m_charges;
    std::vector<double> m_charges_muls;
    dealii::Vector<double> m_total_charge;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    double m_needle_potential = 0, m_bottom_potential = 0;

    const std::string m_name = "Electric_potential";
};

#endif // ELECTRIC_POTENTIAL_HPP_INCLUDED
