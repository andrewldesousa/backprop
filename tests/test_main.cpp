#include "gtest/gtest.h"
#include "../scalar.h"


TEST(ScalarTest, Test1) {
    Scalar<float> s1(1.0);
    Scalar<float> s2(2.0);
    Scalar<float> s4(5.0);
    Scalar<float> s5(10.0);
    Scalar<float>& s3 = s1 + s2;
    Scalar<float>& s6 = (s3 * s4 * s5) - s4;

    s6.grad = 1.0;
    s6.backward();

    // verify the gradient of s1, s2, s3, s4, s5
    EXPECT_FLOAT_EQ(s1.grad, 50.0);
    EXPECT_FLOAT_EQ(s2.grad, 50.0);
    EXPECT_FLOAT_EQ(s3.grad, 50.0);
    EXPECT_FLOAT_EQ(s4.grad, 29.0);
    EXPECT_FLOAT_EQ(s5.grad, 15.0);
}

TEST(ScalarTest, Test2) {
    Scalar<double> s1(1.0);
    Scalar<double> s2(2.0);
    Scalar<double> s3(2.0);
    Scalar<double> s4(5.0);
    Scalar<double> s5(10.0);

    // expression containg all operations
    Scalar<double>& s6 = (s1 + s2) * s3 - s4 * s5;

    s6.grad = 1.0;
    s6.backward();

    // verify the gradient of s1, s2, s3, s4, s5
    EXPECT_DOUBLE_EQ(s1.grad, 2.0);
    EXPECT_DOUBLE_EQ(s2.grad, 2.0);
    EXPECT_DOUBLE_EQ(s3.grad, 3.0);
    EXPECT_DOUBLE_EQ(s4.grad, -10);
    EXPECT_DOUBLE_EQ(s5.grad, -5);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}