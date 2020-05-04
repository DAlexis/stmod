#include "stmod/phys-consts.hpp"

double townsend(double field, double neutral_concentration)
{
    return 1e-21 * field / neutral_concentration;
}
