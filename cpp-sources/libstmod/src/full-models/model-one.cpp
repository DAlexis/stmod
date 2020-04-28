#include "stmod/full-models/model-one.hpp"
#include "stmod/field-output.hpp"
#include "stmod/fe-sampler.hpp"
#include "stmod/fractions.hpp"
#include "stmod/fractions-physics/e.hpp"
#include "stmod/poisson-solver.hpp"
#include "stmod/mesh.hpp"
#include "stmod/poisson-solver-adaptor.hpp"

#include "dsiterpp/time-iter.hpp"
#include "dsiterpp/runge-error-estimator.hpp"
#include "dsiterpp/runge-kutta.hpp"
#include "dsiterpp/euler-explicit.hpp"
#include "stmod/field-output.hpp"

#include <fstream>
#include <sstream>

using namespace dsiterpp;

ModelOne::ModelOne()
{
    init_grid();
    init_poisson_solver();
    init_fractions_storage();
    init_electrons();
    init_time_iterator();
}

void ModelOne::run()
{

}

void ModelOne::output_potential(const std::string& filename)
{

}

void ModelOne::output_fractions(const std::string& filename)
{

}

void ModelOne::output_hook(double real_time)
{
    std::stringstream sstream;
    sstream.setf(std::ios::fixed);
    sstream.precision(10);
    sstream << "t=" << real_time;
    std::string potential_file_name = "potential-" + sstream.str() + ".vtu";
    std::string fractions_file_name = "fractions-" + sstream.str() + ".vtu";
    output_potential(potential_file_name);
    output_fractions(fractions_file_name);
}

void ModelOne::init_grid()
{
    m_grid = std::make_shared<Grid>();
    m_grid->load_from_file(
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.geo",
        "/home/dalexies/Projects/stmod/meshes/spheric-needles-1.msh"
    );
    BoundaryAssigner boundary_assigner(*m_grid);
    boundary_assigner.assign_boundary_ids();
}

void ModelOne::init_poisson_solver()
{

}

void ModelOne::init_fractions_storage()
{

}

void ModelOne::init_electrons()
{
}

void ModelOne::init_time_iterator()
{

}
