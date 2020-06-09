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
    std::cout << "=> Initializing grid" << std::endl;
    model.init_grid();
    std::cout << "=> Initializing fractions" << std::endl;
    model.init_fractions();
    //model.assign_initial_values();
    std::cout << "=> Running" << std::endl;
    model.run();


    return 0;
}
