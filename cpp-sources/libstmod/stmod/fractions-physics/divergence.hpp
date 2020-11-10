#ifndef DIVERGENCE_HPP
#define DIVERGENCE_HPP

#include "stmod/fe-common.hpp"
#include "stmod/fractions/secondary-value.hpp"

class Divergence : public SecondaryValue
{
public:
    Divergence(
            const FEGlobalResources& fe_res,
            const dealii::Vector<double>& field_x,
            const dealii::Vector<double>& field_y,
            const std::string& name);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    // IPreStepComputer
    void compute(double t) override;

private:
    const FEGlobalResources& m_fe_global_res;
    const dealii::Vector<double>& m_field_x;
    const dealii::Vector<double>& m_field_y;

    dealii::Vector<double> m_rhs;
    dealii::Vector<double> m_tmp;


    dealii::SparseDirectUMFPACK m_mass_matrix_inverse;

    dealii::SparseMatrix<double> m_system_matrix;
    dealii::SparseMatrix<double> m_dfield_dx_rhs_matrix;
    dealii::SparseMatrix<double> m_dfield_dy_rhs_matrix;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;
};

#endif // DIVERGENCE_HPP
