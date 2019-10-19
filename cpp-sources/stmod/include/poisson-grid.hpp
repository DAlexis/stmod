#ifndef __POISSON_GRID_HPP__
#define __POISSON_GRID_HPP__

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <string>
#include <map>

class GeometryParameters
{
public:
    void load(const std::string& grid_file_name);

    static std::string trim_string(const std::string& str);
    double needle_rad = 0.0;
    double needle_len = 0.0;
    double cyl_height = 0.0;
    double cyl_rad = 0.0;

private:
    void fill_parameters_map();
    std::map<std::string, double*> m_parameters;
};

class Grid
{
public:
    void load_from_file(const std::string& geo_file, const std::string& msh_name);
    const GeometryParameters& geometry_parameters();
    dealii::Triangulation<2>& triangulation();
    std::vector<std::shared_ptr<dealii::Manifold<2, 2>>>& manifolds();

private:

    dealii::Triangulation<2> m_triangulation;
    GeometryParameters m_geometry_parameters;
    std::vector<std::shared_ptr<dealii::Manifold<2, 2>>> m_manifolds;
};

class BoundaryAssigner
{
public:
    BoundaryAssigner(Grid& grid);
    void assign_boundary_ids();
    void assign_manifold_ids();
private:
    Grid& m_grid;

    constexpr static int near_needle_manifold_id = 2;
};


class AreaConfig
{
public:
    double cylinder_radius = 4;
    double cylinder_height = 4;

    double top_needle_size    = 1;
    double bottom_needle_size = 1;

    double needle_radius    = 0.1;

    unsigned int initial_refine = 3;
    unsigned int rect_needle_refine = 2;
};

class PoissonGrid
{
public:
    constexpr static unsigned int R_AXIS = 0;
    constexpr static unsigned int Z_AXIS = 1;

    constexpr static int BOUNDARY_ID_TOP = 1;
    constexpr static int BOUNDARY_ID_BOTTOM = 2;

    PoissonGrid(const AreaConfig& config);

    void make_grid();

    dealii::Triangulation<2>& triangulation();

private:
    class NeedleTransform
    {
    public:
        NeedleTransform(const AreaConfig& area_conf,
                        double initial_bottom_needle_height,
                        double initial_top_needle_height,
                        double initial_cell_size);

        dealii::Point<2> operator()(const dealii::Point<2>& p);

    private:
        const AreaConfig& m_area_conf;
        double m_initial_bottom_needle_height;
        double m_initial_top_needle_height;
        double m_cell_size;
        double m_gamma;
        double m_coeff;

        double m_h_2;
    };

    void make_initial_grid();
    void needles_remove_cells();
    void assign_boundary_ids(double& out_bottom_needle_height, double& out_top_needle_height);
    void refine_rectangular_needles();
    void transform_needle(double initial_bottom_needle_height,
                          double initial_top_needle_height);

    const AreaConfig& m_config;
    dealii::Triangulation<2> m_triangulation;

    double m_initial_cell_size;
};

#endif // __POISSON_GRID_HPP__
