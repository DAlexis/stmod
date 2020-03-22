#ifndef STMOD_UTILS_HPP_INCLUDED
#define STMOD_UTILS_HPP_INCLUDED

#include <functional>

class LazyInitializer
{
public:
    LazyInitializer(std::function<void(void)>);
};

#endif // STMOD_UTILS_HPP_INCLUDED
