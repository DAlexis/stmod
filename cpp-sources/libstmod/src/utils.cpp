#include "stmod/utils.hpp"

void LazyInitializerBase::clear()
{
    m_is_initialized = false;
}

LazyInitializerCleaner& LazyInitializerCleaner::add(LazyInitializerBase& initializer)
{
    m_initializers.push_back(&initializer);
    return *this;
}

void LazyInitializerCleaner::clear()
{
    for (auto li : m_initializers)
    {
        li->clear();
    }
}
