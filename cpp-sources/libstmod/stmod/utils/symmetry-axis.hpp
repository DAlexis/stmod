#ifndef SYMMETRYAXIS_HPP
#define SYMMETRYAXIS_HPP


#include "stmod/fe-common.hpp"
#include "stmod/grid/grid.hpp"


class AxisRegularizer
{
public:
    AxisRegularizer(const Grid& grid);

    void regularize(dealii::Vector<double>& values, const FEGlobalResources& fe_res);
    std::vector<dealii::types::global_dof_index> get_axis_points();

private:
    const Grid& m_grid;
};


#endif // SYMMETRYAXIS_HPP
