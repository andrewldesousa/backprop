#include <gtest/gtest.h>
#include <memory>
#include "../backprop.h"


TEST(Sigmoid, Test1) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = sigmoid(s1);
    s2->backward();

    EXPECT_FLOAT_EQ(s2->value, 0.7310586);
    EXPECT_FLOAT_EQ(s1->grad, 0.19661193);
}

TEST(CrossEntropy, Test1) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(0.5);
    auto s3 = cross_entropy(s1, s2);
    s3->backward();

    EXPECT_FLOAT_EQ(s3->value, 0.6931472);
    EXPECT_FLOAT_EQ(s2->grad, -2.0);
}
