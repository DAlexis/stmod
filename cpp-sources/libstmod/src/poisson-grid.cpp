#include "stmod/poisson-grid.hpp"

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
    }
}

void GeometryParameters::fill_parameters_map()
{
    m_parameters["needle_rad"] = &needle_rad;
    m_parameters["needle_len"] = &needle_len;
    m_parameters["cyl_height"] = &cyl_height;
    m_parameters["cyl_rad"] = &cyl_rad;
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

BoundaryAssigner::BoundaryAssigner(Grid& grid) :
    m_grid(grid)
{
}

void BoundaryAssigner::assign_boundary_ids()
{
    dealii::Triangulation<2>& tria = m_grid.triangulation();
    const double eps = 0.001;

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
            if (center(0) < eps || center(0) > m_grid.geometry_parameters().cyl_rad - eps)
                continue;

            if (center(1) > m_grid.geometry_parameters().cyl_height / 2.0)
                face->set_boundary_id(1);
            else
                face->set_boundary_id(2);
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

PoissonGrid::PoissonGrid(const AreaConfig& config) :
    m_config(config)
{
    m_initial_cell_size = m_config.cylinder_radius / pow(2, m_config.initial_refine);
}

void PoissonGrid::make_grid()
{
    make_initial_grid();
    needles_remove_cells();

    double bottom_needle_height = 0;
    double top_needle_height = 0;

    assign_boundary_ids(bottom_needle_height, top_needle_height);
    refine_rectangular_needles();
    transform_needle(bottom_needle_height, top_needle_height);

    std::cout << "     Number of active cells: " << m_triangulation.n_active_cells()
                        << std::endl
                        << "     Total number of cells: " << m_triangulation.n_cells()
                        << std::endl;
}

void PoissonGrid::make_initial_grid()
{
    GridGenerator::hyper_rectangle(m_triangulation,
        Point<2>(0, 0),
        Point<2>(m_config.cylinder_radius, m_config.cylinder_height),
        false
    );

    m_triangulation.refine_global(m_config.initial_refine);
}

void PoissonGrid::needles_remove_cells()
{
    using CellIt = typename dealii::Triangulation<2>::active_cell_iterator;
    // Removing cells
    std::set< CellIt > to_remove;
    for (auto it = m_triangulation.begin_active(); it != m_triangulation.end(); ++it)
    {
        if (!it->at_boundary())
            continue;

        // Check cell location
        auto cell_center = it->center();
        if (cell_center(R_AXIS) < m_initial_cell_size
                && (cell_center(Z_AXIS) > m_config.cylinder_height - m_config.top_needle_size
                    || cell_center(Z_AXIS) < m_config.bottom_needle_size)
           )
        {
            to_remove.insert(it);
        }
    }
    //dealii::Triangulation<2> tria_without_cells;

    dealii::GridGenerator::create_triangulation_with_removed_cells(m_triangulation, to_remove, m_triangulation);
    //m_triangulation. = tria_without_cells;
         /*
    std::set< CellIt > toRemoveSet;
    toRemoveSet.insert( dealii::GridTools::find_active_cell_around_point(source, m_pointInCell));
    target.clear();
    dealii::GridGenerator::create_triangulation_with_removed_cells (source, toRemoveSet, target);*/
}

void PoissonGrid::assign_boundary_ids(double& out_bottom_needle_height, double& out_top_needle_height)
{

    for (auto cell = m_triangulation.begin_active(); cell != m_triangulation.end(); ++cell)
    {
        if (!cell->at_boundary())
            continue;

        for (unsigned int face_number=0; face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        {
            auto face = cell->face(face_number);
            if (!face->at_boundary())
            {
                continue;
            }

            Point<2> center = face->center();
            if (center(Z_AXIS) > m_config.cylinder_height - 0.1 * m_initial_cell_size)
            {
                face->set_boundary_id(BOUNDARY_ID_TOP);
            }
            else if (center(Z_AXIS) < 0.1 * m_initial_cell_size)
            {
                face->set_boundary_id(BOUNDARY_ID_BOTTOM);
            }
            else if (center(R_AXIS) > 0.9 * m_initial_cell_size && center(R_AXIS) < 1.1 * m_initial_cell_size)
            {
                if (center(Z_AXIS) > m_config.cylinder_height / 2)
                {
                    face->set_boundary_id(BOUNDARY_ID_TOP);
                } else {
                    face->set_boundary_id(BOUNDARY_ID_BOTTOM);
                }
            }
            else if (center(R_AXIS) > 0.4 * m_initial_cell_size && center(R_AXIS) < 0.6 * m_initial_cell_size)
            {
                if (center(Z_AXIS) > m_config.cylinder_height / 2)
                {
                    face->set_boundary_id(BOUNDARY_ID_TOP);
                    out_top_needle_height = center(Z_AXIS);
                } else {
                    face->set_boundary_id(BOUNDARY_ID_BOTTOM);
                    out_bottom_needle_height = center(Z_AXIS);
                }
            }
        }
    }
}

void PoissonGrid::refine_rectangular_needles()
{
    m_triangulation.refine_global(m_config.rect_needle_refine);
}

void PoissonGrid::transform_needle(double initial_bottom_needle_height,
                                   double initial_top_needle_height)
{
    NeedleTransform transformation_func(
        m_config,
        initial_bottom_needle_height,
        initial_top_needle_height,
        m_initial_cell_size
    );
    GridTools::transform(
        [&transformation_func](const dealii::Point<2>& p)
        {
            return transformation_func(p);
        },
        m_triangulation
    );

    /*
    std::map<unsigned int, Point<2>> border_move;


    for (auto face = m_triangulation.begin_active(); face != m_triangulation.end(); ++face)
    {
        for (unsigned int i=0; i < GeometryInfo<2>::vertices_per_face; i++)
        {
            unsigned int ind = face->vertex_index(i);
            Point<2> position = m_triangulation.get_vertices()[ind];
            auto new_position = transformation_func(position);

            border_move[ind] = new_position;
        }
    }
    GridTools::laplace_transform(border_move, m_triangulation);*/
}

dealii::Triangulation<2>& PoissonGrid::triangulation()
{
    return m_triangulation;
}

PoissonGrid::NeedleTransform::NeedleTransform(
        const AreaConfig& area_conf,
        double initial_bottom_needle_height,
        double initial_top_needle_height,
        double initial_cell_size
) :
    m_area_conf(area_conf),
    m_initial_bottom_needle_height(initial_bottom_needle_height),
    m_initial_top_needle_height(initial_top_needle_height),
    m_cell_size(initial_cell_size),
    m_h_2(m_area_conf.cylinder_height / 2)
{
    m_gamma = log(m_area_conf.needle_radius / m_area_conf.cylinder_radius) / log(m_cell_size / m_area_conf.cylinder_radius);
    m_coeff = pow(m_area_conf.cylinder_radius, 1 - m_gamma);
}

dealii::Point<2> PoissonGrid::NeedleTransform::operator()(const dealii::Point<2>& p)
{
    double r = p(0);
    if (r > m_cell_size)
        r = m_coeff * pow(p(0), m_gamma);
    else
        r *= m_area_conf.needle_radius / m_cell_size;
    double z = p(1);
    if (r < m_area_conf.needle_radius)
    {
        if (z < m_h_2 && z >= m_initial_bottom_needle_height)
        {
            double dz_current = (m_h_2 - z);
            double vertical_space_fixed = (m_h_2 - (m_area_conf.bottom_needle_size + sqrt(pow(m_area_conf.needle_radius, 2) - pow(r, 2))) );
            double vertical_space_orig = (m_h_2 - m_initial_bottom_needle_height);
            double dz_fixed = dz_current * vertical_space_fixed / vertical_space_orig;
            z = m_h_2 - dz_fixed;
        }
        else if (z > m_h_2 && z <= m_initial_top_needle_height)
        {
            double dz_current = (z - m_h_2);
            double vertical_space_fixed = (m_h_2 - (m_area_conf.top_needle_size + sqrt(pow(m_area_conf.needle_radius, 2) - pow(r, 2))) );
            double vertical_space_orig = (m_initial_top_needle_height - m_h_2);
            double dz_fixed = dz_current * vertical_space_fixed / vertical_space_orig;
            z = m_h_2 + dz_fixed;
        }
    }

    // Fixing shape
    double rho = sqrt(pow(r, 2) + pow(z - m_area_conf.bottom_needle_size, 2) );
    double dist_to_needle = rho - m_area_conf.needle_radius;
    double dist_to_needle_limit = 4 * m_area_conf.needle_radius;

    if (z < m_h_2 && z > m_area_conf.bottom_needle_size && dist_to_needle > 0.0)
    {
        if (dist_to_needle > dist_to_needle_limit)
            dist_to_needle = dist_to_needle_limit;
        /*
        double r_factor = 1 - r / (4 * m_area_conf.needle_radius);
        if (r_factor < 0.0)
            r_factor = 0.0;




        if (dist_to_needle > 4 * m_area_conf.needle_radius)
            dist_to_needle = 4 * m_area_conf.needle_radius;

        if (dist_to_needle > 0) // && rho < 4 * m_area_conf.needle_radius)
        {
            double rho_factor = dist_to_needle / (2 * m_area_conf.needle_radius);
            rho_factor = rho_factor > 1.0 ? 1.0 : rho_factor;

            r += r_factor * dist_to_needle * rho_factor * r * 2;
        }*/
    }

    return Point<2>(r, z);
}
