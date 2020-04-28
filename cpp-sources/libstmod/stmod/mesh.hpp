#ifndef MESH_HPP_INCLUDED
#define MESH_HPP_INCLUDED

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

class GeometryParameters
{
public:
    void load(const std::string& grid_file_name);

    static std::string trim_string(const std::string& str);
    double needle_rad = 0.0;
    double needle_len = 0.0;
    double cyl_height = 0.0;
    double cyl_rad = 0.0;
    double scale = 0.0;

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

    void debug_make_rectangular();

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

    constexpr static dealii::types::boundary_id top_and_needle = 1;
    constexpr static dealii::types::boundary_id bottom = 2;
    constexpr static dealii::types::boundary_id outer_border = 3;
    constexpr static dealii::types::boundary_id axis = 4;

private:
    Grid& m_grid;

    constexpr static int near_needle_manifold_id = 2;
};


#endif // MESH_HPP_INCLUDED
