#include "stmod/tensors.hpp"

#include <algorithm>

FullTensor3::FullTensor3(unsigned int size_i, unsigned int size_j, unsigned int size_k) :
    m_size_i(size_i), m_size_j(size_j), m_size_k(size_k)
{
    m_content.resize(m_size_i * m_size_j * m_size_k, 0.0);
}

double& FullTensor3::operator()(unsigned int i, unsigned int j, unsigned int k)
{
#ifdef DEBUG
    if (i >= m_size_i || j >= m_size_j || k >= m_size_k)
        throw std::range_error("FullTensor error: index out of range");
#endif
    return m_content[k + j * m_size_k + i * m_size_k * m_size_j];
}

void FullTensor3::operator=(double x)
{
    std::fill(m_content.begin(), m_content.end(), x);
}

void SparseTensor3::clear()
{
    m_unordered_map.clear();
    m_nonzero.clear();
}

double SparseTensor3::operator()(SparseTensor3::IndexType i, SparseTensor3::IndexType j, SparseTensor3::IndexType k) const
{
    auto it = m_unordered_map.find(i);
    if (it == m_unordered_map.end())
        return 0.0;

    auto jt = it->second.find(j);
    if (jt == it->second.end())
        return 0.0;

    auto kt = jt->second.find(k);
    if (kt == jt->second.end())
        return 0.0;

    return kt->second;
}

void SparseTensor3::set(SparseTensor3::IndexType i, SparseTensor3::IndexType j, SparseTensor3::IndexType k, double value)
{
    m_unordered_map[i][j][k] = value;
    m_nonzero.push_back(IndexesTuple(i, j, k));
}

const std::vector<SparseTensor3::IndexesTuple>& SparseTensor3::nonzero() const
{
    return m_nonzero;
}


void SparseTensor3::sum_with_tensor(dealii::Vector<double>& out_vector,
                     const dealii::Vector<double>& in_first,
                     const dealii::Vector<double>& in_second) const
{
    for (auto & indexes : nonzero())
    {
        SparseTensor3::IndexType i = std::get<0>(indexes);
        SparseTensor3::IndexType j = std::get<1>(indexes);
        SparseTensor3::IndexType k = std::get<2>(indexes);
        out_vector[k] += (*this)(i, j, k) * in_first[i] * in_second[j];
    }
}

void SparseTensor3::sum_with_tensor(dealii::SparseMatrix<double>& out_matrix,
                     const dealii::Vector<double>& vector,
                     IndexType axis) const
{
    if (axis != 0)
        throw std::range_error("Sum along axis != 0 not supported, but easy to implement");

    for (auto & indexes : nonzero())
    {
        SparseTensor3::IndexType i = std::get<0>(indexes);
        SparseTensor3::IndexType j = std::get<1>(indexes);
        SparseTensor3::IndexType k = std::get<2>(indexes);
        // The order is k, j because we constuct tensor with third index as row index,
        // but matrixes are multiplied from left, so they have first index as row index
        out_matrix(k, j) += (*this)(i, j, k) * vector[i];
    }
}
