#ifndef TENSORS_HPP_INCLUDED
#define TENSORS_HPP_INCLUDED

#include <unordered_map>
#include <vector>

struct FullTensor3
{
public:
    FullTensor3(unsigned int size_i, unsigned int size_j, unsigned int size_k);

    double& operator()(unsigned int i, unsigned int j, unsigned int k);
    void operator=(double x);

private:
    unsigned int m_size_i, m_size_j, m_size_k;

    std::vector<double> m_content;
};


struct SparseTensor3
{
public:
    using IndexType = unsigned int;
    using IndexesTuple = std::tuple<IndexType, IndexType, IndexType>;

    void clear();

    double operator()(IndexType i, IndexType j, IndexType k) const;
    void set(IndexType i, IndexType j, IndexType  k, double value);
    const std::vector<IndexesTuple>& nonzero() const;

private:
    std::unordered_map<IndexType, std::unordered_map<IndexType, std::unordered_map<IndexType, double>>> m_unordered_map;
    std::vector<IndexesTuple> m_nonzero;
};


#endif // TENSORS_HPP_INCLUDED
