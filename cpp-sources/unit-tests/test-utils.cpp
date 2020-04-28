#include "stmod/utils.hpp"

#include "gtest/gtest.h"


TEST(LazyInitializer, Usage)
{
    LazyInitializer<int> i1([](int& v) { v = 5; } );
    ASSERT_EQ(i1.get(), 5);
    i1.get() = 6;
    ASSERT_EQ(i1.get(), 6);

    i1.clear();
    ASSERT_EQ(i1.get(), 5);
    i1 = 7;

    int p = i1;
    ASSERT_EQ(p, 7);

    LazyInitializer<int> i2([](int& v) { v = -1; } );

    LazyInitializerCleaner cleaner;
    cleaner.add(i1).add(i2);

    ASSERT_EQ(i1.get(), 7);
    ASSERT_EQ(i2.get(), -1);

    i2.get() = -2;
    ASSERT_EQ(i2.get(), -2);

    cleaner.clear();
    ASSERT_EQ(i1.get(), 5);
    ASSERT_EQ(i2.get(), -1);

    struct TS
    {
        int a = 0;
        int b = 1;
    };

    LazyInitializer<TS> its([](TS& ts) { ts = TS(); });
    ASSERT_EQ(its->a, 0);
    ASSERT_EQ(its->b, 1);

    its->a = 3;
    ASSERT_EQ(its->a, 3);
    its.clear();
    ASSERT_EQ(its->a, 0);
}
