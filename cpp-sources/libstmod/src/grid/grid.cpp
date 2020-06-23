#include "stmod/grid/grid.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace dealii;

void GeometryParameters::load(const std::string& grid_file_name)
{
    fill_parameters_map();

    std::ifstream geo_file(grid_file_name.c_str());
    std::string line;
    while (std::getline(geo_file, line))
    {
        size_t pos = line.find("=");
        if (pos == line.npos)
            continue;
        try {
            std::string key = trim_string(line.substr(0, pos));
            std::string value = trim_string(line.substr(pos+1, line.size()));
            auto it = m_parameters.find(key);
            if (it != m_parameters.end())
            {
                *(it->second) = std::stod(value);
            }
        } catch (std::exception&)
        {
        }
    }
    for (auto & it : m_parameters)
    {
        if (*(it.second) == 0.0)
            throw std::range_error("Parameter " + it.first + " is not set in file " + grid_file_name);

        *(it.second) *= scale;
    }
}

void GeometryParameters::fill_parameters_map()
{
    m_parameters["needle_rad"] = &needle_rad;
    m_parameters["needle_len"] = &needle_len;
    m_parameters["cyl_height"] = &cyl_height;
    m_parameters["cyl_rad"] = &cyl_rad;
    m_parameters["scale"] = &scale;
}

std::string GeometryParameters::trim_string(const std::string& str)
{
    std::string result;
    if (str.empty())
        return result;
    size_t begin = 0, end = str.size()-1;
    while(std::isspace(str[begin]) && begin < str.size())
        begin++;

    while(std::isspace(str[end]) && end > 0)
        end--;

    result = str.substr(begin, end+1);
    return result;
}

void Grid::load_from_file(const std::string& geo_file, const std::string& msh_name)
{
    dealii::GridIn<2> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(msh_name.c_str());
    gridin.read_msh(f);

    m_geometry_parameters.load(geo_file);
}

const GeometryParameters& Grid::geometry_parameters()
{
    return m_geometry_parameters;
}

dealii::Triangulation<2>& Grid::triangulation()
{
    return m_triangulation;
}

std::vector<std::shared_ptr<dealii::Manifold<2, 2>>>& Grid::manifolds()
{
    return m_manifolds;
}

void Grid::debug_make_rectangular()
{
    m_triangulation.clear();
    GridGenerator::hyper_cube(m_triangulation, 0, 1);
    GridTools::transform(
            [this](const dealii::Point<2>& p)
            {
                return dealii::Point<2>(p[0]*m_geometry_parameters.cyl_rad, p[1]*m_geometry_parameters.cyl_height);
            },
            m_triangulation
        );
    m_triangulation.refine_global(5);
}

BoundaryAssigner::BoundaryAssigner(Grid& grid) :
    m_grid(grid)
{
}

void BoundaryAssigner::assign_boundary_ids()
{
    dealii::Triangulation<2>& tria = m_grid.triangulation();
    const double eps = 0.000001;

    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
        if (!cell->at_boundary())
            continue;

        for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (!face->at_boundary())
                continue;

            auto center = face->center();

            if (center(0) > m_grid.geometry_parameters().cyl_rad - eps)
            {
                face->set_boundary_id(outer_border);
                continue;
            }

            if (center(0) < eps)
            {
                face->set_boundary_id(axis);
                continue;
            }

            if (center(1) > m_grid.geometry_parameters().cyl_height / 2.0)
            {
                face->set_boundary_id(top_and_needle);
                continue;
            }

            face->set_boundary_id(bottom);
        }
    }
}

void BoundaryAssigner::assign_manifold_ids()
{
    dealii::Triangulation<2>& tria = m_grid.triangulation();
    const double needle_center_z = m_grid.geometry_parameters().cyl_height - m_grid.geometry_parameters().needle_len;
    const double manifold_radius = m_grid.geometry_parameters().needle_rad * 10;

    const Point<2> needle_center(0, needle_center_z);

    auto manifolds = m_grid.manifolds();
    auto spherical_manifold = std::make_shared<SphericalManifold<2>> (needle_center);
    manifolds.push_back(spherical_manifold);

    m_grid.triangulation().set_manifold(near_needle_manifold_id, *spherical_manifold);

    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
        auto center = cell->center();
        double d = needle_center.distance(center);
        if (center[1] < needle_center[1] && d < manifold_radius)
        {
            cell->set_manifold_id(near_needle_manifold_id);
        }
    }
}
