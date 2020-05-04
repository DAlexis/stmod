#include "stmod/fractions-physics/reaction-constants.hpp"
#include <cmath>


double k_5a(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);

    if (td < 300)
    {
        return pow(10, -14.09-402.9/td);
    } else {
        return pow(10, -13.37-618.1/td);
    }
}

double k_5b(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);

    if (td < 260)
    {
        return pow(10, -14.31-285.7/td);
    } else {
        return (1 + 4e-10 * pow(td, 3)) * pow(10, -13.54-485.7/td);
    }
}


double k_1a(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);

    if (td < 90)
    {
        return pow(10, -15.42-127/td);
    } else {
        return pow(10, -16.21-57/td);
    }
}

double k_1b(double T_e, double T)
{
    return 1.4e-41 * (300 / T_e) * exp(-600 / T) * exp(700 * (T_e - T) / (T_e * T));
}

double k_1c(double T_e, double T)
{
    return 1.07e-43 * pow((300 / T_e), 2) * exp(-70 / T) * exp(1500 * (T_e - T) / (T_e * T));
}

double k_6(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);
    return 1.6e-18 * exp(- pow(48.9/(11 + td), 2) );
}

double k_7(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);
    return 1.24e-17 * exp(- pow(179/(8.8 + td), 2) );
}

double k_2(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);
    return 6.96e-17 * exp(- pow(198/(5.6 + td), 2) );
}

double k_3(double field, double neutral_concentration)
{
    double td = townsend(field, neutral_concentration);
    return 1.1e-42 * exp(-pow(td/65, 2));
}

double k_4(double T)
{
    return 3.5e-31 * (300 / T);
}

double k_10(double T)
{
    return 1e-10 * exp(-1044/T);
}
