#include "stmod/full-models/model-one.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <fstream>

using namespace dealii;
using namespace std;

int main()
{
    ModelOne model;
    model.init_grid();
    model.init_fractions();
    model.assign_initial_values();
    model.run();


    return 0;
}
