#include "stmod/full-models/model-one.hpp"

#include "stmod/fe-sampler.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fractions/fraction.hpp"
#include "stmod/field-operating/gradient.hpp"
#include "stmod/field-operating/l2norm.hpp"
#include "stmod/field-operating/divergence.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/fractions-physics/electrons-flow.hpp"
#include "stmod/fractions-physics/electric-potential.hpp"
#include "stmod/output/output.hpp"
#include "stmod/time/time-iteration.hpp"
#include "stmod/grid/mesh-refiner.hpp"
#include "stmod/phys-consts.hpp"
#include "stmod/utils/symmetry-axis.hpp"

#include "stmod/fe-common.hpp"
#include "stmod/grid/grid.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <sstream>

class NeInitialFunction : public dealii::Function<2>
{
public:
    double value(const dealii::Point<2> &p, const unsigned int component = 0) const override
    {
        double result = 1e15;
        const double y_decay = 0.001;
//        const double y_limit = 0.01475;
        const double y_limit = 0.016;
        if (p[1] < 2 * y_decay && p[1] > y_decay)
        {
            result *= (p[1] - y_decay) / y_decay;
        }
        if (p[1] < y_decay)
        {
            result = 0.0;
        }

        if (p[1] > y_limit - y_decay && p[1] < y_limit)
        {
            result *= 1.0 - (p[1] - (y_limit - y_decay)) / y_decay;
        }

        if (p[1] > y_limit)
        {
            result = 0.0;
        }


        return result;
    }
};

ModelOne::ModelOne()
{
}

ModelOne::~ModelOne()
{
}

void ModelOne::init_grid()
{
    m_grid.load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    //grid.debug_make_rectangular();
    m_boundary_assigner.reset(new BoundaryAssigner(m_grid));
    m_boundary_assigner->assign_boundary_ids();

    m_global_resources.reset(new FEGlobalResources(m_grid.triangulation(), 1));
    m_global_resources->on_triangulation_updated();

    m_refiner.reset(new MeshRefiner(*m_global_resources));

    m_variables_collector.reset(new VariablesCollector(m_global_resources->constraints()));
}

void ModelOne::init_fractions()
{
    m_electrons_flow_parameters.reset(new ElectronsFlowParameters);
    m_electrons_flow_parameters->mu = 5.92;
    m_electrons_flow_parameters->D = 0.1;

    // Creating fractrions pointers
    m_Ne.reset(new Electrons(*m_global_resources));
    m_refiner->add_mesh_based(m_Ne.get());
    m_variables_collector->add_derivatives_provider(m_Ne.get());
    m_output_maker.add(m_Ne.get());

    add_fraction(m_O_minus, new Fraction("O_minus"));
    add_fraction(m_O_2_minus, new Fraction ("O_2_minus"));
    add_fraction(m_O_3_minus, new Fraction ("O_3_minus"));
    add_fraction(m_O_4_minus, new Fraction ("O_4_minus"));
    add_fraction(m_N_p, new Fraction ("N_p"));
    add_fraction(m_O, new Fraction ("O"));
    add_fraction(m_u, new Fraction ("u"));
    add_fraction(m_v, new Fraction ("v"));
    add_fraction(m_w, new Fraction ("w"));
    add_fraction(m_x, new Fraction ("x"));
    add_fraction(m_y, new Fraction ("y"));
    add_fraction(m_z, new Fraction ("z"));
    add_fraction(m_W_v, new Fraction ("W_v"));
    add_fraction(m_O_3, new Fraction ("O_3"));
    add_fraction(m_T, new Fraction ("T"));

    std::cout << "   Fractions added" << std::endl;

    // Electric field
    add_secondary(m_electric_potential, new ElectricPotential (*m_global_resources));
    add_secondary(m_E_x, new Gradient (*m_global_resources, *m_electric_potential, "E_x", 0, -1.0));
    add_secondary(m_E_y, new Gradient (*m_global_resources, *m_electric_potential, "E_y", 1, -1.0));
    add_secondary(m_E_norm, new L2Norm(*m_E_x, *m_E_y, "E_norm"));



    // Fundamental components
    add_secondary(m_M, new SecondaryFunction("M",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = m_T->values()[i];
            return secondary_M(T);
        }
    ));

    add_secondary(m_N_2, new SecondaryFunction("N_2",
        [this](dealii::types::global_dof_index i, double)
        {
            return 0.8*(*m_M)[i];
        }
    ));

    add_secondary(m_O_2, new SecondaryFunction("O_2",
        [this](dealii::types::global_dof_index i, double)
        {
            return 0.2*(*m_M)[i];
        }
    ));


    add_secondary(m_Td, new SecondaryFunction("Td",
        [this](dealii::types::global_dof_index i, double)
        {
            double E_scalar = m_E_norm->values()[i];
            return 1e21 * E_scalar / (*m_M)[i];
        }
    ));

    add_secondary(m_Te, new SecondaryFunction("Te",
        [this](dealii::types::global_dof_index i, double)
        {
            double Td = (*m_Td)[i];
            if (Td < 50)
            {
                return 0.447 * pow(Td, 0.16);
            } else {
                return 0.0167 * Td;
            }
        }
    ));

    // Electrons movement
    add_secondary(m_electrons_diffusion_flow_x, new Gradient (*m_global_resources, *m_Ne, "diffusion_flow_x", 0, -1));
    add_secondary(m_electrons_diffusion_flow_y, new Gradient (*m_global_resources, *m_Ne, "diffusion_flow_y", 1, -1));

    add_secondary(m_electrons_drift_speed, new SecondaryFunction("drift_speed",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            return 0.25 * pow(td, 0.864) * 1e6 * 1e-2;
        }
    ));

    // Electrons drift flow
    add_secondary(m_electrons_drift_flow_x, new SecondaryFunction("drift_flow_x",
        [this](dealii::types::global_dof_index i, double)
        {
            double v = (*m_electrons_drift_speed)[i];
            double Ne = (*m_Ne)[i];
            double Ex = (*m_E_x)[i];
            double E = (*m_E_norm)[i];
            return -v * Ne * Ex / E;
        }
    ));

    add_secondary(m_electrons_drift_flow_y, new SecondaryFunction("drift_flow_y",
        [this](dealii::types::global_dof_index i, double)
        {
            double v = (*m_electrons_drift_speed)[i];
            double Ne = (*m_Ne)[i];
            double Ey = (*m_E_y)[i];
            double E = (*m_E_norm)[i];
            return -v * Ne * Ey / E;
        }
    ));

    // Electrons total flow
    add_secondary(m_total_electrons_flow_x, new SecondaryFunction("total_flow_x",
        [this](dealii::types::global_dof_index i, double)
        {
            double drift_x = (*m_electrons_drift_flow_x)[i];
            double diff_x = (*m_electrons_diffusion_flow_x)[i];
            return drift_x + diff_x;
        }
    ));

    add_secondary(m_total_electrons_flow_y, new SecondaryFunction("total_flow_y",
        [this](dealii::types::global_dof_index i, double)
        {
            double drift_y = (*m_electrons_drift_flow_y)[i];
            double diff_y = (*m_electrons_diffusion_flow_y)[i];
            return drift_y + diff_y;
        }
    ));

//    add_secondary(m_electrons_flow_x, new ElectronsFlow(*m_global_resources, *m_Ne, *m_E_x, 0, *m_electrons_flow_parameters));
//    add_secondary(m_electrons_flow_y, new ElectronsFlow(*m_global_resources, *m_Ne, *m_E_y, 1, *m_electrons_flow_parameters));

    //add_secondary(m_div_J, new Divergence(*m_global_resources, *m_electrons_flow_x, *m_electrons_flow_y, "elec_flow_divergence"));
    add_secondary(m_div_J, new Divergence(*m_global_resources, *m_total_electrons_flow_x, *m_total_electrons_flow_y, "elec_flow_divergence"));

    // Heat power
    add_secondary(m_heat_power, new SecondaryFunction("heat_power",
        [this](dealii::types::global_dof_index i, double)
        {
            double Ne = m_Ne->values()[i];
            double E = m_E_norm->values()[i];
            return 1.6e-19 * (5.92 * Ne) * pow(E, 2);
        }
    ));

    // Reaction coefficients
    add_secondary(m_k_1, new SecondaryFunction("k1",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 100)
            {
                return pow(10, -14.4-171.1/td);
            } else {
                return pow(10, -14.67-144/td);
            }
        }
    ), false);

    add_secondary(m_k_2, new SecondaryFunction("k2",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 100)
            {
                return pow(10, -13.91-168.1/td);
            } else {
                return pow(10, -14.2-139.2/td);
            }
        }
    ), false);

    add_secondary(m_k_3, new SecondaryFunction("k3",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 100)
            {
                return pow(10, -14.17-187.4/td);
            } else {
                return pow(10, -14.29-175.3/td);
            }
        }
    ), false);

    add_secondary(m_k_4, new SecondaryFunction("k4",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 200)
            {
                return pow(10, -13.88-233.2/td);
            } else {
                return pow(10, -14.08-193.7/td);
            }
        }
    ), false);

    add_secondary(m_k_6, new SecondaryFunction("k6",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 300)
            {
                return pow(10, -14.09-402.9/td);
            } else {
                return pow(10, -13.37-618.1/td);
            }
        }
    ), false);


    add_secondary(m_k_7, new SecondaryFunction("k7",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 260)
            {
                return pow(10, -14.31-285.7/td);
            } else {
                return (1 + 4e-10 * pow(td, 3)) * pow(10, -13.54-485.7/td);
            }
        }
    ), false);

    add_secondary(m_k_8, new SecondaryFunction("k8",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 100)
            {
                return pow(10, -13.78-140.8/td);
            } else if (td < 300) {
                return pow(10, -14.31-87.8/td);
                return pow(10, -14.31-87.8/td);
            } else {
                return pow(10, -14.6);
            }
        }
    ), false);

    add_secondary(m_k_9, new SecondaryFunction("k9",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 100)
            {
                return pow(10, -13.43-170.6/td);
            } else {
                return pow(10, -13.6-154.3/td);
            }
        }
    ), false);

    add_secondary(m_k_10, new SecondaryFunction("k10",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            if (td < 90)
            {
                return pow(10, -15.42-127/td);
            } else {
                return pow(10, -16.21-57/td);
            }
        }
    ), false);

    add_secondary(m_k_11, new SecondaryFunction("k11",
        [this](dealii::types::global_dof_index i, double)
        {
            double Te = (*m_Te)[i];
            double T = (*m_T)[i];
            return 1.4e-41 * (300 / Te) * exp(-600 / T) * exp(700 * (Te - T) / (Te * T));
        }
    ), false);

    add_secondary(m_k_12, new SecondaryFunction("k12",
        [this](dealii::types::global_dof_index i, double)
        {
            double Te = (*m_Te)[i];
            double T = (*m_T)[i];
            return 1.07e-43 * pow((300 / Te), 2) * exp(-70 / T) * exp(1500 * (Te - T) / (Te * T));
        }
    ), false);

    add_secondary(m_k_13, new SecondaryFunction("k13",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            return 1.16e-18 * exp(- pow(48.9/(11 + td), 2) );
        }
    ), false);

    add_secondary(m_k_14, new SecondaryFunction("k14",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            return 1.24e-17 * exp(- pow(179/(8.8 + td), 2) );
        }
    ), false);

    add_secondary(m_k_15, new SecondaryConstant("k15", 3e-16), false);

    add_secondary(m_k_16, new SecondaryFunction("k16",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            return 6.96e-17 * exp(- pow(198/(5.6 + td), 2) );
        }
    ), false);

    add_secondary(m_k_17, new SecondaryFunction("k17",
        [this](dealii::types::global_dof_index i, double)
        {
            double td = (*m_Td)[i];
            return 1.1e-42 * exp(- pow(td/65, 2) );
        }
    ), false);

    add_secondary(m_k_18, new SecondaryConstant("k18", 3.2e-16));

    add_secondary(m_k_19, new SecondaryFunction("k19",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 3.5e-43 * (300 / T);
        }
    ), false);

    add_secondary(m_k_20, new SecondaryFunction("k20",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 1e-16 * exp(-1044 / T);
        }
    ), false);

    add_secondary(m_k_21, new SecondaryConstant("k21", 4e-16), false);

    add_secondary(m_k_100, new SecondaryFunction("k100",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 6e-46 * pow(300 / T, 2.0);
        }
    ), true);

    add_secondary(m_k_101, new SecondaryFunction("k101",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 1.8e-17 * exp(107 / T);
        }
    ), false);

    add_secondary(m_k_102, new SecondaryFunction("k102",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 3.2e-17 * exp(67 / T);
        }
    ), false);

    add_secondary(m_k_103, new SecondaryFunction("k103",
        [this](dealii::types::global_dof_index i, double)
        {
            return 2*(*m_k_8)[i] + (*m_k_9)[i];
        }
    ), false);

    add_secondary(m_k_104, new SecondaryFunction("k104",
        [this](dealii::types::global_dof_index i, double)
        {
            return 1 / (4.83e-23 * (*m_M)[i]);
        }
    ), false);

    add_secondary(m_k_105, new SecondaryFunction("k105",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_104)[i] * (1 - (*m_f_v)[i]);
        }
    ), false);

    add_secondary(m_k_106, new SecondaryFunction("k106",
        [this](dealii::types::global_dof_index i, double)
        {
            return 11594 / (*m_M)[i];
        }
    ), false);

    add_secondary(m_k_107, new SecondaryFunction("k107",
        [this](dealii::types::global_dof_index i, double)
        {
            double T = (*m_T)[i];
            return 2.484e-17 * exp(107 / T);
        }
    ), false);

    add_secondary(m_k_diss_eff, new SecondaryFunction("k_diss_eff",
        [this](dealii::types::global_dof_index i, double)
        {
            double result = 4 * ((*m_k_1)[i] + (*m_k_2)[i] + (*m_k_3)[i] + (*m_k_4)[i]) + (*m_k_6)[i] + (*m_k_7)[i] + (*m_k_8)[i];
            //double result = 4 * ((*m_k_1)[i] + (*m_k_2)[i] + (*m_k_3)[i] + (*m_k_4)[i] + (*m_k_5)[i]);
            return result;
        }
    ), true);

    add_secondary(m_beta_ep, new SecondaryFunction("beta_ep",
        [this](dealii::types::global_dof_index i, double)
        {
            double Te = (*m_Te)[i];
            return 2e-12 * pow((300.0 / Te), 0.5);
        }
    ), false);

    add_secondary(m_beta_np, new SecondaryConstant("beta_np", 1e-13));

    add_secondary(m_f_v, new SecondaryFunction("f_v",
        [this](dealii::types::global_dof_index i, double)
        {
            // TODO: use tabulated func
            return 0.5;
        }
    ), false);

    add_secondary(m_L, new SecondaryFunction("W(T)",
        [this](dealii::types::global_dof_index i, double)
        {
            double M = (*m_M)[i];
            double T = (*m_T)[i];
            return secondary_L(T, M);
        }
    ));

    add_secondary(m_Q, new SecondaryFunction("Q",
        [this](dealii::types::global_dof_index i, double)
        {
            double W_v = (*m_W_v)[i];
            double L = (*m_L)[i];
            double O = (*m_O)[i];
            double O_3 = (*m_O_3)[i];

            return (W_v - L) * (4.5e-21 * O + 5e-20 * O_3);
        }
    ));


    // DEBUG SECONDARIES
    // Negative components
    add_secondary(m_k10_O2_Ne, new SecondaryFunction("k10_O2_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_10)[i] * (*m_O_2)[i] * (*m_Ne)[i];
        }
    ));

    add_secondary(m_k11_O2_O2_Ne, new SecondaryFunction("k11_O2_O2_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_11)[i] * (*m_O_2)[i] * (*m_O_2)[i] * (*m_Ne)[i];
        }
    ));

    add_secondary(m_k12_O2_N2_Ne, new SecondaryFunction("k12_O2_N2_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_12)[i] * (*m_O_2)[i] * (*m_N_2)[i] * (*m_Ne)[i];
        }
    ));

    add_secondary(m_beta_Np_Ne, new SecondaryFunction("beta_Np_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_beta_np)[i] * (*m_N_p)[i] * (*m_Ne)[i];
        }
    ));


    // Positive components
    add_secondary(m_k6_N2_Ne, new SecondaryFunction("k6_N2_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_6)[i] * (*m_N_2)[i] * (*m_Ne)[i];
        }
    ));

    add_secondary(m_k7_O2_Ne, new SecondaryFunction("k7_O2_Ne",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_7)[i] * (*m_O_2)[i] * (*m_Ne)[i];
        }
    ));

    add_secondary(m_k13_N2_O_minus, new SecondaryFunction("k13_N2_O_minus",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_13)[i] * (*m_N_2)[i] * (*m_O_minus)[i];
        }
    ));

    add_secondary(m_k14_M_O_2_minus, new SecondaryFunction("k14_M_O_2_minus",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_14)[i] * (*m_M)[i] * (*m_O_2_minus)[i];
        }
    ));

    add_secondary(m_k15_O_O_3_minus, new SecondaryFunction("k15_O_O_3_minus",
        [this](dealii::types::global_dof_index i, double)
        {
            return (*m_k_15)[i] * (*m_O)[i] * (*m_O_3_minus)[i];
        }
    ));


    std::cout << "   Secondary values added" << std::endl;

    m_refiner->call_on_mesh_refine();

    std::cout << "   Mesh-dependent initialized" << std::endl;

    // Equations
    // Ne
    (*m_Ne)
         .add_source(-1.0, *m_div_J)
         .add_source(1.0, *m_k_6, *m_N_2, *m_Ne)
         .add_source(1.0, *m_k_7, *m_O_2, *m_Ne)
         .add_source(-1.0, *m_k_10, *m_O_2, *m_Ne)
         .add_source(-1.0, *m_k_11, *m_O_2, *m_O_2, *m_Ne)
         .add_source(-1.0, *m_k_12, *m_O_2, *m_N_2, *m_Ne)
         .add_source(1.0, *m_k_13, *m_N_2, *m_O_minus)
         .add_source(1.0, *m_k_14, *m_M, *m_O_2_minus)
         .add_source(1.0, *m_k_15, *m_O, *m_O_3_minus)
         .add_source(-1.0, *m_beta_ep, *m_Ne, *m_N_p)
         //.add_source(2.0e-18, *m_O_2) // cosmic rays ionization - ???
    ;

    dealii::VectorTools::interpolate(m_global_resources->dof_handler(), NeInitialFunction(), m_Ne->values_w());

    //m_Ne->apply_boundary_to_concentration();

    //*m_Ne = initial_Ne;
    //assign_test_initial_values();



    // O_minus
    (*m_O_minus)
        .add_source(1.0, *m_k_10, *m_O_2, *m_Ne)
        .add_source(-1.0, *m_k_13, *m_N_2, *m_O_minus)
        .add_source(-1.0, *m_k_16, *m_O_2, *m_O_minus)
        .add_source(-1.0, *m_k_17, *m_M, *m_O_2, *m_O_minus)
        .add_source(1.0, *m_k_15, *m_O, *m_O_4_minus)
        .add_source(-1.0, *m_beta_np, *m_O_minus, *m_N_p)
    ;

    *m_O_minus = 0.0;

    // O_2_minus
    (*m_O_2_minus)
        .add_source(1.0, *m_k_11, *m_O_2, *m_O_2, *m_Ne)
        .add_source(1.0, *m_k_12, *m_O_2, *m_N_2, *m_Ne)
        .add_source(1.0, *m_k_16, *m_O_2, *m_O_minus)
        .add_source(-1.0, *m_k_14, *m_M, *m_O_2_minus)
        .add_source(1.0, *m_k_18, *m_O, *m_O_3_minus)
        .add_source(-1.0, *m_k_19, *m_O_2, *m_M, *m_O_2_minus)
        .add_source(1.0, *m_k_20, *m_M, *m_O_4_minus)
        .add_source(-1.0, *m_beta_np, *m_O_2_minus, *m_N_p)
    ;

    *m_O_2_minus = 0.0;

    // O_3_minus
    (*m_O_3_minus)
        .add_source(1.0, *m_k_17, *m_M, *m_O_2_minus, *m_O_minus)
        .add_source(-1.0, *m_k_15, *m_O, *m_O_3_minus)
        .add_source(-1.0, *m_k_18, *m_O, *m_O_3_minus)
        .add_source(1.0, *m_k_21, *m_O, *m_O_4_minus)
        .add_source(-1.0, *m_beta_np, *m_O_3_minus, *m_N_p)
    ;

    *m_O_3_minus = 0.0;

    // O_4_minus
    (*m_O_4_minus)
        .add_source(1.0, *m_k_19, *m_O_2, *m_M, *m_O_2_minus)
        .add_source(-1.0, *m_k_20, *m_M, *m_O_4_minus)
        .add_source(-1.0, *m_k_21, *m_O, *m_O_4_minus)
        .add_source(-1.0, *m_k_15, *m_O, *m_O_4_minus)
        .add_source(-1.0, *m_beta_np, *m_O_4_minus, *m_N_p)
    ;

    *m_O_4_minus = 0.0;

    // HERE
    // O
    (*m_O)
       .add_source(2.0, *m_k_diss_eff, *m_Ne, *m_O_2)
       .add_source(-1.0, *m_k_100, *m_O, *m_O_2, *m_M) // THIS is too fast, ~1e-16
    ;

    *m_O = 0.0;

    // O_3
    (*m_O_3)
        .add_source(6.2e-40, *m_O, *m_O_2, *m_M)
    ;

    *m_O_3 = 0.0;

    // N_p
    (*m_N_p)
        .add_source(1.0, *m_k_6, *m_N_2, *m_Ne)
        .add_source(1.0, *m_k_7, *m_O_2, *m_Ne)
        .add_source(-1.0, *m_beta_ep, *m_Ne, *m_N_p)
        .add_source(-1.0, *m_beta_np, *m_O_minus, *m_N_p)
        .add_source(-1.0, *m_beta_np, *m_O_2_minus, *m_N_p)
        .add_source(-1.0, *m_beta_np, *m_O_3_minus, *m_N_p)
        .add_source(-1.0, *m_beta_np, *m_O_4_minus, *m_N_p)
        //.add_source(2.0e-18, *m_O_2) // cosmic rays ionization - ???
    ;

    //*m_N_p = initial_Ne;
    dealii::VectorTools::interpolate(m_global_resources->dof_handler(), NeInitialFunction(), m_N_p->values_w());

    // u
    (*m_u)
        .add_source(1.0, *m_k_1, *m_Ne, *m_N_2)
        .add_source(2e-19, *m_N_2, *m_v)
        .add_source(-(1.7e-18 + 7.5e-19), *m_O_2, *m_u)
        .add_source(23.7e-17, *m_u, *m_u)
        .add_source(3e-17, *m_u, *m_z)
    ;

    *m_u = 0.0;

    // v
    (*m_v)
        .add_source(1.0, *m_k_2, *m_Ne, *m_N_2)
        .add_source(7.7e-17, *m_u, *m_u)
        .add_source(2e-19, *m_N_2, *m_w)
        .add_source(1e-17, *m_N_2, *m_x)
        .add_source(2.4e7, *m_x)
        .add_source(-3e-16, *m_O_2, *m_v)
        .add_source(-2e-19, *m_N_2, *m_v)
    ;

    *m_v = 0.0;

    // w
    (*m_w)
        .add_source(1.0, *m_k_3, *m_Ne, *m_N_2)
        .add_source(-2.8e-17, *m_O_2, *m_w)
        .add_source(-2e-19, *m_N_2, *m_w)
    ;

    *m_w = 0.0;

    // x
    (*m_x)
        .add_source(1.0, *m_k_4, *m_Ne, *m_N_2)
        .add_source(1.6e-16, *m_u, *m_u)
        .add_source(-2.5e-16, *m_O_2, *m_x)
        .add_source(-1e-17, *m_N_2, *m_x)
        .add_source(-2.4e7, *m_x)
    ;

    *m_x = 0.0;

    // y
    (*m_y)
        .add_source(1.0, *m_k_9, *m_Ne, *m_O_2)
        .add_source(2.8e-17, *m_O_2, *m_w)
        .add_source(-1.0, *m_k_101, *m_N_2, *m_y)
        .add_source(-1.0, *m_k_102, *m_O_2, *m_y)
    ;

    *m_y = 0.0;

    // z
    (*m_z)
        .add_source(1.0, *m_k_103, *m_Ne, *m_O_2)
        .add_source(1.0, *m_k_101, *m_N_2, *m_y)
        .add_source(1.0, *m_k_102, *m_O_2, *m_y)
        .add_source(3.4e-18, *m_O_2, *m_u)
        .add_source(6e-16, *m_O_2, *m_v)
        .add_source(2.8e-17, *m_O_2, *m_w)
        .add_source(3e-17, *m_u, *m_z)
    ;

    *m_z = 0.0;

    // W_v
    (*m_W_v)
        .add_source(1.0, *m_f_v, *m_heat_power)
        .add_source(-1.0, *m_Q)
    ;

    double initial_T = 300;
    *m_W_v = secondary_L(initial_T, secondary_M(initial_T));



    // m_T
    (*m_T)
        .add_source(1.0, *m_k_105, *m_heat_power)
        .add_source(1.0, *m_k_104, *m_Q)
        .add_source(1.7e-18, *m_k_106, *m_u, *m_O_2)
        .add_source(3.08e-16, *m_k_106, *m_u, *m_u)
        .add_source(7.05e-16, *m_k_106, *m_v, *m_O_2)
        .add_source(3.92e-17, *m_k_106, *m_w, *m_O_2)
        .add_source(1.2075e-15, *m_k_106, *m_x, *m_O_2)
        .add_source(1.0, *m_k_106, *m_k_107, *m_y, *m_N_2)
    ;

    *m_T = initial_T;

    std::cout << "   Fractions equations initialized" << std::endl;

    m_electric_potential->add_charge(m_Ne->values(), - Consts::e);
    m_electric_potential->add_charge(m_O_minus->values(), -Consts::e);
    m_electric_potential->add_charge(m_O_2_minus->values(), -Consts::e);
    m_electric_potential->add_charge(m_O_3_minus->values(), -Consts::e);
    m_electric_potential->add_charge(m_O_4_minus->values(), -Consts::e);
    m_electric_potential->add_charge(m_N_p->values(), Consts::e);

    //m_Ne->set_electric_field(*m_E_x, *m_E_y, m_electric_potential->total_chagre());

    //m_variables_collector->add_implicit_steppable(m_Ne.get());
    std::cout << "   All fractions setup done" << std::endl;
}

void ModelOne::assign_test_initial_values()
{
    FieldAssigner fa(m_global_resources->dof_handler());
    fa.assign_fiend(
        m_Ne->values_w(),
        [](const dealii::Point<2>& point) -> double
        {
            return 1e13*exp(- (pow((point[0]) / 0.002, 2.0) + pow((point[1] - 0.012 ) / 0.005, 2.0)));
        }
    );
}

void ModelOne::run()
{
    StmodTimeStepper stepper;

    //m_output_maker.output(m_global_resources->dof_handler(), "frac-out-iter-" + std::to_string(0) + ".vtu");
    //m_refiner->do_refine();
    //m_refiner->do_refine();
    //boundary_assigner.assign_boundary_ids();
    //m_electric_potential->compute(0.0);

    std::cout << "   Computing electric potential and performing first 'Dry' output" << std::endl;
    m_electric_potential->compute(0.0);

    const auto fname = make_output_filename(0);
    std::cout << "Writing " << fname << std::endl;
    m_output_maker.output(m_global_resources->dof_handler(), fname);

    //return 0.0;

    double t = 0;
    double dt = 1e-13;
    double last_output_t = t;

    for(int i = 0; !m_interrupt; i++)
    {
        /*m_electrons->compute(0.0);
        m_electrons->value_w().add(0.00000005, m_electrons->derivative());
        */
        std::cout << "# Number of dofs: " << m_global_resources->n_dofs() << std::endl;
        double t_new = stepper.iterate(*m_variables_collector, t, dt);
        m_Ne->apply_cathode_supression();
        //m_refiner->do_refine(m_Ne->values());
        dt = t_new - t;
        t = t_new;
/*
        m_boundary_assigner->assign_boundary_ids();
        m_electric_potential->compute(0.0);
        std::cout << "Iteration " << i << "; t = " << t << "... " << std::flush;
        auto & dn = m_electrons->get_implicit_delta(dt, 0.5);

        m_electrons->values_w().add(1.0, dn);
        remove_negative(m_electrons->values_w());
        m_global_resources->constraints().distribute(m_electrons->values_w());*/
        //double dt = stepper.iterate(var_coll, t, 5e-9);
/*
        double t_new = stepper.iterate(var_coll, t, dt);
        dt = t_new - t;
        t = t_new;
        std::cout << "dt = " << dt << std::endl;
*/
        AxisRegularizer axis_regularizer(m_grid);
        axis_regularizer.regularize(m_Ne->values_w(), *m_global_resources);
        if (t - last_output_t >= 1e-11 || i == 0)
        {
            const auto fname = make_output_filename(t);
            std::cout << "Writing " << fname << std::endl;
            m_output_maker.output(m_global_resources->dof_handler(), fname);
            last_output_t = t;
        }

        /*if (i % 15 == 0)
            m_refiner->do_refine(m_Ne->values());*/
    }
}

void ModelOne::interrupt()
{
    m_interrupt = true;
}

std::string ModelOne::make_output_filename(double t)
{
    std::ostringstream filename;
    filename << "frac-out-iter-" << std::setw(15) << std::setfill('0') << size_t(t * 1e15) << ".vtu";
    return filename.str();
}

void ModelOne::register_secondary(SecondaryValue* value, bool need_output)
{
    m_refiner->add_mesh_based(value);
    m_variables_collector->add_pre_step_computator(value);
    if (need_output)
        m_output_maker.add(value);
}

void ModelOne::add_fraction(std::unique_ptr<Fraction>& uniq_ptr, Fraction* fraction)
{
    uniq_ptr.reset(fraction);
    m_refiner->add_mesh_based(uniq_ptr.get());
    m_variables_collector->add_derivatives_provider(uniq_ptr.get());
    m_output_maker.add(uniq_ptr.get());
}

double ModelOne::secondary_L(double T, double M)
{
    return M * 4.64e-20 / (exp(3362 / T) - 1);
}

double ModelOne::secondary_M(double T)
{
    return 2.7e25 * (((760))) / 760 * 273 / T;
}

