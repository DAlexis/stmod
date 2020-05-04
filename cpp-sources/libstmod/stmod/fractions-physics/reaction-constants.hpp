#ifndef REACTION_CONSTANTS_HPP_INCLUDED
#define REACTION_CONSTANTS_HPP_INCLUDED

#include "stmod/phys-consts.hpp"

double k_5a(double field, double neutral_concentration);
double k_5b(double field, double neutral_concentration);
double k_1a(double field, double neutral_concentration);
double k_1b(double T_e, double T);
double k_1c(double T_e, double T);
double k_6(double field, double neutral_concentration);
double k_7(double field, double neutral_concentration);
constexpr static double k_8 = 3e-16;
double k_2(double field, double neutral_concentration);
double k_3(double field, double neutral_concentration);
double k_4(double T);
constexpr static double k_9 = 3.2e-16;
double k_10(double T);
constexpr static double k_11 = 4e-10;
constexpr static double k_12 = 3e-10;

#endif // REACTION_CONSTANTS_HPP_INCLUDED
