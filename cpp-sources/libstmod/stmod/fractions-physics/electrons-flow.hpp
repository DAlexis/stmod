#ifndef ELECTRONSFLOW_HPP
#define ELECTRONSFLOW_HPP

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

struct ElectronsFlowParameters
{
    double mu = 0;
    double D = 0;
};

class ElectronsFlow : public SecondaryValue
{
public:
    ElectronsFlow(
            const FEGlobalResources& fe_res,
            const dealii::Vector<double>& concentration,
            const dealii::Vector<double>& E_field_component,
            size_t component,
            ElectronsFlowParameters flow_parameters);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepComputer
    void compute(double t) override;

    void set_electric_field(const dealii::Vector<double>& E_component);
    void set_electrons_density(const dealii::Vector<double>& n_e);


private:
    const FEGlobalResources& m_fe_global_res;
    const ElectronsFlowParameters m_flow_parameters;

    const dealii::Vector<double>& m_n_e;
    const dealii::Vector<double>& m_E_component;

    double m_component;
    dealii::Vector<double> m_rhs;

    dealii::SparseMatrix<double> m_tmp_matrix;
    dealii::SparseMatrix<double> m_system_matrix;
    dealii::SparseMatrix<double> m_rhs_matrix;
    dealii::SparseDirectUMFPACK m_system_matrix_inverse;

    std::map<dealii::types::global_dof_index, double> m_boundary_values; // Empty
};

#endif // ELECTRONSFLOW_HPP
