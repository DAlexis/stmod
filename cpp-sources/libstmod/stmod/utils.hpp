#ifndef STMOD_UTILS_HPP_INCLUDED
#define STMOD_UTILS_HPP_INCLUDED

#include <functional>
#include <vector>

class LazyInitializerBase
{
public:
    void clear();

protected:
    bool m_is_initialized = false;
};

class LazyInitializerCleaner
{
public:
    LazyInitializerCleaner& add(LazyInitializerBase& initializer);
    void clear();

private:
    std::vector<LazyInitializerBase*> m_initializers;
};

template <typename ResourceType>
class LazyInitializer : public LazyInitializerBase
{
public:
    using InitializerType = std::function<void(ResourceType&)>;

    LazyInitializer(InitializerType initializer = nullptr) :
        m_initializer(initializer)
    {
    }

    LazyInitializer(InitializerType initializer, LazyInitializerCleaner& cleaner) :
        m_initializer(initializer)
    {
        cleaner.add(*this);
    }

    LazyInitializer(LazyInitializerCleaner& cleaner) :
        m_initializer(nullptr)
    {
        cleaner.add(*this);
    }

    LazyInitializer(const LazyInitializer&) = delete;

    void set_initializer(InitializerType initializer)
    {
        m_initializer = initializer;
    }

    ResourceType& get()
    {
        if (!m_is_initialized)
        {
            m_initializer(m_resource);
            m_is_initialized = true;
        }
        return m_resource;
    }

    const ResourceType& get() const
    {
        return const_cast<LazyInitializer<ResourceType>*>(this)->get();
    }

    operator ResourceType& () { return get(); }
    operator const ResourceType& () const { return get(); }

    ResourceType* operator->() { return &get(); }
    const ResourceType* operator->() const { return &get(); }

    void operator=(const ResourceType& right) { get() = right; }

private:
    ResourceType m_resource;

    InitializerType m_initializer;
};

#endif // STMOD_UTILS_HPP_INCLUDED
