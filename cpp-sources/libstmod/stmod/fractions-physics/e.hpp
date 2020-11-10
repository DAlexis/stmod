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

class Electrons : public Fraction
{
public:
    Electrons(const FEGlobalResources& fe_res);

    // IMeshBased
    void init_mesh_dependent(const dealii::DoFHandler<2>& dof_handler) override;

    void compute_derivatives(double t) override;

    void apply_cathode_supression();

    Fraction& operator=(double value);

private:
    class CathodeSupressionFunction : public dealii::Function<2>
    {
    public:
        CathodeSupressionFunction(double scale);
        double value(const dealii::Point<2> &p, const unsigned int component = 0) const override;

    private:
        double m_scale;
    };
    using PairSourceTuple = std::tuple<const dealii::Vector<double>*, const dealii::Vector<double>*>;

    void create_cathode_supression_field();

    const FEGlobalResources& m_fe_global_res;

    dealii::Vector<double> m_system_rhs;
    dealii::Vector<double> m_cathode_supression_field;

    std::map<dealii::types::global_dof_index, double> m_boundary_values;

    static const std::string m_names[2];
};

#endif // FRACTIONS_PHYSICS_E_HPP_INCLUDED
