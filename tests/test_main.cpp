#include "gtest/gtest.h"
#include "../scalar.h"


TEST(ScalarAddition, Test1) {
    // shared pointer to a scalar object
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0);
    auto s3 = s1 + s2;

    EXPECT_FLOAT_EQ(s3->value, 3.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, 1.0);
}

// TEST(ScalarAddition, Test2) {
//     Scalar<double> s1(1.0);
//     Scalar<double>& s2 = 1 + s1;
//     EXPECT_DOUBLE_EQ(s2.value, 2.0);

//     s2.backward();
//     EXPECT_DOUBLE_EQ(s1.grad, 1.0);
// }

TEST(ScalarSubtraction, Test1) {
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0); 
    auto s3 = s1 - s2;

    EXPECT_FLOAT_EQ(s3->value, -1.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, -1.0);
}

TEST(ScalarMultiplication, Test1) {
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0);
    auto s3 = s1 * s2;

    EXPECT_FLOAT_EQ(s3->value, 2.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 2.0);
    EXPECT_FLOAT_EQ(s2->grad, 1.0);
}

TEST(ScalarDivision, Test1) {
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0);
    auto s3 = s1 / s2;

    EXPECT_FLOAT_EQ(s3->value, 0.5);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 0.5);
    EXPECT_FLOAT_EQ(s2->grad, -0.25);
}

// // TEST(ScalarDivision, Test2) {
// //     Scalar<double> s1(1.0);
// //     Scalar<double>& s2 = 2 / s1;
// //     EXPECT_DOUBLE_EQ(s2.value, 2.0);

// //     s2.backward();
// //     EXPECT_DOUBLE_EQ(s1.grad, -2.0);
// // }

TEST(ScalarMixedOperationsTest, Test1) {
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0);
    auto s3 = std::make_shared<Scalar<float>>(2.0);
    auto s4 = std::make_shared<Scalar<float>>(5.0);
    auto s5 = std::make_shared<Scalar<float>>(10.0);

    auto s6 = (s1 + s2) * s3 - s4 * s5;

    s6->backward();
    EXPECT_FLOAT_EQ(s1->grad, 2.0);
    EXPECT_FLOAT_EQ(s2->grad, 2.0);
    EXPECT_FLOAT_EQ(s3->grad, 3.0);
    EXPECT_FLOAT_EQ(s4->grad, -10.0);
    EXPECT_FLOAT_EQ(s5->grad, -5.0);
}

// TEST(ScalarMixedOperationsTest, Test2) {
//     auto s1 = std::make_shared<Scalar<float>>(1.0);
//     auto s2 = std::make_shared<Scalar<float>>(2.0);
//     auto s3 = std::make_shared<Scalar<float>>(2.0);
//     auto s4 = std::make_shared<Scalar<float>>(5.0);
//     auto s5 = std::make_shared<Scalar<float>>(10.0);

//     auto s6 = (s1 + s2) * (s3 - s4) / s5;

//     s6->backward();
//     EXPECT_FLOAT_EQ(s1->grad, 0.1);
//     EXPECT_FLOAT_EQ(s2->grad, 0.1);
//     EXPECT_FLOAT_EQ(s3->grad, 0.1);
//     EXPECT_FLOAT_EQ(s4->grad, -0.02);
//     EXPECT_FLOAT_EQ(s5->grad, -0.02);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}