#ifndef FULL_MODELS_MODEL_ONE_HPP_INCLUDED
#define FULL_MODELS_MODEL_ONE_HPP_INCLUDED

#include "stmod/grid/grid.hpp"
#include "stmod/output/output.hpp"

class FEGlobalResources;
class MeshRefiner;

class ElectricPotential;
class Electrons;
class VariablesCollector;
class Fraction;
class SecondaryValue;
class SecondaryConstant;
class SecondaryFunction;
class ElectronsFlowParameters;
class ElectronsFlow;
class Gradient;
class Divergence;
class L2Norm;

class ModelOne
{
public:
    ModelOne();
    ~ModelOne();

    void init_grid();
    void init_fractions();
    void assign_test_initial_values();

    void run();
    void interrupt();

private:
    std::string make_output_filename(double t);
    template<typename UniqPtrType, typename PtrType>
    void add_secondary(std::unique_ptr<UniqPtrType>& uniq_ptr, PtrType* value, bool need_output = true)
    {
        uniq_ptr.reset(value);
        register_secondary(uniq_ptr.get());
    }
    void register_secondary(SecondaryValue* value, bool need_output = true);
    void add_fraction(std::unique_ptr<Fraction>& uniq_ptr, Fraction* fraction);

    static double secondary_L(double T, double M);
    static double secondary_M(double T);

    bool m_interrupt = false;

    Grid m_grid;
    OutputMaker m_output_maker;

    std::unique_ptr<BoundaryAssigner> m_boundary_assigner;
    std::unique_ptr<FEGlobalResources> m_global_resources;
    std::unique_ptr<MeshRefiner> m_refiner;
    std::unique_ptr<VariablesCollector> m_variables_collector;

    std::unique_ptr<ElectronsFlowParameters> m_electrons_flow_parameters;
    std::unique_ptr<Divergence> m_div_J;

    // Main secondary functions
    std::unique_ptr<ElectricPotential> m_electric_potential;
    std::unique_ptr<Gradient> m_E_x;
    std::unique_ptr<Gradient> m_E_y;
    std::unique_ptr<L2Norm> m_E_norm;
    std::unique_ptr<ElectronsFlow> m_electrons_flow_x;
    std::unique_ptr<ElectronsFlow> m_electrons_flow_y;
    //std::unique_ptr<HeatPower> m_heat_power_2;

    std::unique_ptr<SecondaryValue> m_M;
    std::unique_ptr<SecondaryValue> m_N_2;
    std::unique_ptr<SecondaryValue> m_O_2;
    std::unique_ptr<SecondaryValue> m_Te;
    std::unique_ptr<SecondaryValue> m_Td;
    std::unique_ptr<SecondaryValue> m_f_v;
    std::unique_ptr<SecondaryValue> m_heat_power;
    std::unique_ptr<SecondaryValue> m_L;


    // Additional secondary functions
    std::unique_ptr<SecondaryValue> m_k_1;
    std::unique_ptr<SecondaryValue> m_k_2;
    std::unique_ptr<SecondaryValue> m_k_3;
    std::unique_ptr<SecondaryValue> m_k_4;
//    std::unique_ptr<SecondaryValue> m_k_5;
    std::unique_ptr<SecondaryValue> m_k_6;
    std::unique_ptr<SecondaryValue> m_k_7;
    std::unique_ptr<SecondaryValue> m_k_8;
    std::unique_ptr<SecondaryValue> m_k_9;
    std::unique_ptr<SecondaryValue> m_k_10;
    std::unique_ptr<SecondaryValue> m_k_11;
    std::unique_ptr<SecondaryValue> m_k_12;
    std::unique_ptr<SecondaryValue> m_k_13;
    std::unique_ptr<SecondaryValue> m_k_14;
    std::unique_ptr<SecondaryValue> m_k_15;
    std::unique_ptr<SecondaryValue> m_k_16;
    std::unique_ptr<SecondaryValue> m_k_17;
    std::unique_ptr<SecondaryValue> m_k_18;
    std::unique_ptr<SecondaryValue> m_k_19;
    std::unique_ptr<SecondaryValue> m_k_20;
    std::unique_ptr<SecondaryValue> m_k_21;
    std::unique_ptr<SecondaryValue> m_k_100;
    std::unique_ptr<SecondaryValue> m_k_101;
    std::unique_ptr<SecondaryValue> m_k_102;
    std::unique_ptr<SecondaryValue> m_k_103;
    std::unique_ptr<SecondaryValue> m_k_104;
    std::unique_ptr<SecondaryValue> m_k_105;
    std::unique_ptr<SecondaryValue> m_k_106;
    std::unique_ptr<SecondaryValue> m_k_107;
    std::unique_ptr<SecondaryValue> m_k_diss_eff;
    std::unique_ptr<SecondaryValue> m_beta_ep;
    std::unique_ptr<SecondaryValue> m_beta_np;
    std::unique_ptr<SecondaryValue> m_Q;

    // Fractions
    std::unique_ptr<Electrons> m_Ne;
    std::unique_ptr<Fraction> m_O_minus;
    std::unique_ptr<Fraction> m_O_2_minus;
    std::unique_ptr<Fraction> m_O_3_minus;
    std::unique_ptr<Fraction> m_O_4_minus;
    std::unique_ptr<Fraction> m_N_p;
    std::unique_ptr<Fraction> m_O;
    std::unique_ptr<Fraction> m_u;
    std::unique_ptr<Fraction> m_v;
    std::unique_ptr<Fraction> m_w;
    std::unique_ptr<Fraction> m_x;
    std::unique_ptr<Fraction> m_y;
    std::unique_ptr<Fraction> m_z;
    std::unique_ptr<Fraction> m_W_v;
    std::unique_ptr<Fraction> m_O_3;
    std::unique_ptr<Fraction> m_T;
};

#endif // FULL_MODELS_MODEL_ONE_HPP_INCLUDED
