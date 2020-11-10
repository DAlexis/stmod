#ifndef FE_COMMON_HPP_INCLUDED
#define FE_COMMON_HPP_INCLUDED

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

void remove_negative(dealii::Vector<double>& values);

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
    const dealii::SparseMatrix<double>& r_grad_phi_i_comp_phi_j(size_t component) const;

    const SparseTensor3& grad_phi_i_grad_phi_j_dot_r_phi_k() const;

    const dealii::SparseDirectUMFPACK& inverse_mass_matrix() const;

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
    LazyInitializer<dealii::SparseMatrix<double>> r_grad_phi_i_0_phi_j{m_cleaner};
    LazyInitializer<dealii::SparseMatrix<double>> r_grad_phi_i_1_phi_j{m_cleaner};
    LazyInitializer<dealii::SparseDirectUMFPACK> m_inverse_mass_matrix{m_cleaner};

    LazyInitializer<SparseTensor3> m_grad_phi_i_grad_phi_j_dot_r_phi_k{m_cleaner};
};

#endif // FE_COMMON_HPP_INCLUDED
