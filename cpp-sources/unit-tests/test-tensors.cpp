#include "stmod/tensors.hpp"

#include "gtest/gtest.h"


TEST(SparseTensor3, Basics)
{
    SparseTensor3 t;
    t.set(2, 3, 4, 3.14);
    ASSERT_NEAR(t(2, 3, 4), 3.14, 1e-6);
    ASSERT_NEAR(t(98676, 232, 400000), 0.0, 1e-6);
}
