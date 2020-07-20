#include "stmod/full-models/model-one.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <fstream>
#include <signal.h>

using namespace dealii;
using namespace std;

std::unique_ptr<ModelOne> model;

void signal_handler(int signum)
{
    constexpr static int hits_limit = 5;
    static int hits_count = 0;
    cout << "Interrupt signal (" << signum << ") received" << endl;
    model->interrupt();
    if (signum == SIGINT)
    {
        if (++hits_count == hits_limit)
        {
            cout << "OK, interrupting" << endl;
            exit(SIGINT);
        }
        cout << "To kill immediately, hit " << hits_limit - hits_count << " Ctrl+C times more :)" << endl;
    }
}

int main()
{
    model.reset(new ModelOne());
    signal(SIGINT, signal_handler);
    std::cout << "=> Initializing grid" << std::endl;
    model->init_grid();
    std::cout << "=> Initializing fractions" << std::endl;
    model->init_fractions();
    //model.assign_initial_values();
    std::cout << "=> Running" << std::endl;
    model->run();
    return 0;
}
