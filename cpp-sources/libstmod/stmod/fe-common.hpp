#ifndef FE_COMMON_HPP_INCLUDED
#define FE_COMMON_HPP_INCLUDED

#include "stmod/lin-eq-solver.hpp"
#include "stmod/tensors.hpp"
#include "stmod/utils.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <functional>
/*
class FEResourcesNew
{
public:
    FEResources(dealii::Triangulation<2>& triangulation, dealii::AffineConstraints<double>& constraints, unsigned int degree);

    dealii::Triangulation<2>& triangulation();

    void init();

private:
    dealii::AffineConstraints<double> m_constraints;
    LazyInitializerCleaner m_cleaner;
};*/

class IFEGlobalResourcesUser
{
public:
    virtual void on_triangulation_updated() = 0;
    virtual ~IFEGlobalResourcesUser() = default;
};

class FEGlobalResources
{
public:
    FEGlobalResources(dealii::Triangulation<2>& triangulation, unsigned int degree);

    dealii::Triangulation<2>& triangulation();
    void on_triangulation_updated();

    unsigned int degree() const;
    const dealii::FE_Q<2>& fe() const;
    const dealii::DoFHandler<2>& dof_handler() const;
    dealii::types::global_dof_index n_dofs() const;
    const dealii::SparsityPattern& sparsity_pattern() const;
    const dealii::AffineConstraints<double>& constraints() const;

    const dealii::SparseMatrix<double>& mass_matrix() const;
    const dealii::SparseMatrix<double>& laplace_matrix() const;

    void add_subscriber(IFEGlobalResourcesUser* subscriber);

private:
    dealii::Triangulation<2>& m_triangulation;
    dealii::FE_Q<2> m_fe;
    dealii::DoFHandler<2> m_dof_handler;

    std::vector<IFEGlobalResourcesUser*> m_subscribers;

    dealii::SparsityPattern m_sparsity_pattern;
    dealii::AffineConstraints<double> m_constraints;

    LazyInitializerCleaner m_cleaner;
    LazyInitializer<dealii::SparseMatrix<double>> m_laplace_matrix{m_cleaner};
    LazyInitializer<dealii::SparseMatrix<double>> m_mass_matrix{m_cleaner};
};

class FEResources : public IFEGlobalResourcesUser
{
public:
    using BoundaryConditionsGeneratorFunc = std::function<void(dealii::AffineConstraints<double>&)>;

    FEResources(FEGlobalResources& global_resources);
    dealii::AffineConstraints<double>& constraints();

    void set_boundary_cond_gen(BoundaryConditionsGeneratorFunc gen);

    // IFEGlobalResourcesUser
    void on_triangulation_updated() override;

    const FEGlobalResources& global_resources() const;

    /**
     * @brief Matrix for (d/dr(psi_i), psi_j) + (grad psi_i, grad(r * psi_j)),
     * so this is matrix for [r * laplacian_in_axial_coordinates]
     */
    const dealii::SparseMatrix<double>& r_laplace_matrix_axial() const;
    /**
     * @brief Matrix for (psi_i, r*psi_j)
     *
     */
    const dealii::SparseMatrix<double>& r_mass_matrix() const;

    const dealii::SparseDirectUMFPACK& inverse_r_mass_matrix() const;
    const dealii::SparseDirectUMFPACK& inverse_r_laplace_matrix() const;

    const SparseTensor3& phi_i_phi_j_dot_r_phi_k() const;
    const SparseTensor3& grad_phi_i_grad_phi_j_dot_r_phi_k() const;

    const dealii::SparseMatrix<double>& mass_matrix() const;
    const dealii::SparsityPattern& sparsity_pattern() const;
    const dealii::AffineConstraints<double>& constraints() const;
    const LinEqSolver& lin_eq_solver() const;

private:

    const FEGlobalResources& m_global_resources;
    BoundaryConditionsGeneratorFunc m_boundary_conditions_maker_func = nullptr;

    LazyInitializerCleaner m_cleaner;

    dealii::AffineConstraints<double> m_constraints;

    LinEqSolver m_lin_eq_solver{m_constraints, 1e-2};

    dealii::SparsityPattern  m_sparsity_pattern;

    LazyInitializer<dealii::SparseMatrix<double>> m_r_laplace_matrix{m_cleaner};
    LazyInitializer<dealii::SparseMatrix<double>> m_r_mass_matrix{m_cleaner};

    LazyInitializer<dealii::SparseDirectUMFPACK> m_inverse_r_mass_matrix{m_cleaner};
    LazyInitializer<dealii::SparseDirectUMFPACK> m_inverse_r_laplace_matrix{m_cleaner};

    LazyInitializer<SparseTensor3> m_phi_i_phi_j_dot_r_phi_k;
    LazyInitializer<SparseTensor3> m_grad_phi_i_grad_phi_j_dot_r_phi_k;
};

#endif // FE_COMMON_HPP_INCLUDED
