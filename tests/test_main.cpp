#include "gtest/gtest.h"
#include "../scalar.h"


TEST(ScalarAddition, Test1) {
    // shared pointer to a scalar object
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(2.0);
    auto s3 = s1 + s2;

    EXPECT_FLOAT_EQ(s3->value, 3.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, 1.0);
}

TEST(ScalarSubtraction, Test1) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(2.0); 
    auto s3 = s1 - s2;

    EXPECT_FLOAT_EQ(s3->value, -1.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, -1.0);
}

TEST(ScalarMultiplication, Test1) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(2.0);
    auto s3 = s1 * s2;

    EXPECT_FLOAT_EQ(s3->value, 2.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 2.0);
    EXPECT_FLOAT_EQ(s2->grad, 1.0);
}

TEST(ScalarDivision, Test1) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(2.0);
    auto s3 = s1 / s2;

    EXPECT_FLOAT_EQ(s3->value, 0.5);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 0.5);
    EXPECT_FLOAT_EQ(s2->grad, -0.25);
}

TEST(ScalarDivision, Test2) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(1.0);
    auto s3 = s1 / s2;

    EXPECT_FLOAT_EQ(s3->value, 1.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, -1.0);
}

TEST(ScalarDivision, Test3) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = Scalar<float>::make(0.5);
    auto s3 = s1 / s2;

    EXPECT_FLOAT_EQ(s3->value, 2.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 2);
    EXPECT_FLOAT_EQ(s2->grad, -4.0);
}

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

TEST(ScalarMixedOperationsTest, Test2) {
    auto s1 = Scalar<float>::make(1.0);
    auto s2 = std::make_shared<Scalar<float>>(2.0);
    auto s3 = std::make_shared<Scalar<float>>(2.0);
    auto s4 = std::make_shared<Scalar<float>>(5.0);
    auto s5 = std::make_shared<Scalar<float>>(10.0);

    auto s6 = (s1 - s2) * s3 + s4 * s5;

    s6->backward();
    EXPECT_FLOAT_EQ(s1->grad, s3->value);
    EXPECT_FLOAT_EQ(s2->grad, s3->value * -1.0);
    EXPECT_FLOAT_EQ(s3->grad, s1->value - s2->value);
    EXPECT_FLOAT_EQ(s4->grad, s5->value);
    EXPECT_FLOAT_EQ(s5->grad, s4->value);
}

TEST(ScalarMixedOperationsTest, Test3) {
    auto s1 = Scalar<double>::make(1.0);
    auto s2 = Scalar<double>::make(2.0);
    auto s3 = Scalar<double>::make(2.0);

    auto s4 = (s1 - s2);
    auto s5 = s4 * s3;

    s5->backward();

    EXPECT_FLOAT_EQ(s1->grad, 2.0);
    EXPECT_FLOAT_EQ(s2->grad, -2.0);
    EXPECT_FLOAT_EQ(s3->grad, -1.0);
}

TEST(ScalarMixedOperationsTest, Test4) {
    auto x = Scalar<float>::make(1.0);
    auto negative_x = -x;
    auto exp_negative_x = exp(negative_x);
    auto numerator = Scalar<float>::make(1);
    auto denominator_one = Scalar<float>::make(1);
    auto denominator = denominator_one + exp_negative_x;
    auto result = numerator / denominator;
    result->backward();

    // verify the results
    EXPECT_FLOAT_EQ(result->value, 0.7310586);
    EXPECT_FLOAT_EQ(x->grad, 0.19661193);
}


TEST(ScalarNegative, Test1) {
    auto s1 = Scalar<double>::make(1.0);
    auto s2 = -s1;

    EXPECT_FLOAT_EQ(s2->value, -1.0);

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, -1.0);
}

TEST(ScalarNegative, Test2) {
    auto s1 = Scalar<double>::make(1.0);
    auto s2 = -s1;
    auto s3 = -s2;

    EXPECT_FLOAT_EQ(s2->value, -1.0);
    EXPECT_FLOAT_EQ(s3->value, 1.0);

    s3->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
    EXPECT_FLOAT_EQ(s2->grad, -1.0);
}

TEST(ScalarPlus, Test1) {
    auto s1 = Scalar<double>::make(1.0);
    auto s2 = +s1;

    EXPECT_FLOAT_EQ(s2->value, 1.0);

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
}

TEST(ScalarExp, Test1) {
    auto s1 = Scalar<double>::make(1.0);
    auto s2 = exp(s1);

    EXPECT_FLOAT_EQ(s2->value, exp(1.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, exp(1.0));
}

TEST(ScalarExp, Test2) {
    auto s1 = Scalar<double>::make(0.0);
    auto s2 = exp(s1);

    EXPECT_FLOAT_EQ(s2->value, exp(0.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, exp(0.0));
}

TEST(ScalarExp, Test3) {
    auto s1 = Scalar<double>::make(-1.0);
    auto s2 = exp(s1);

    EXPECT_FLOAT_EQ(s2->value, exp(-1.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, exp(-1.0));
}

TEST(ScalarExp, Test4) {
    auto s1 = Scalar<double>::make(-2.0);
    auto s2 = exp(s1);

    EXPECT_FLOAT_EQ(s2->value, exp(-2.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, exp(-2.0));
}

TEST(ScalarLog, Test1) {
    auto s1 = std::make_shared<Scalar<float>>(1.0);
    auto s2 = log(s1);

    EXPECT_FLOAT_EQ(s2->value, log(1.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, 1.0);
}

TEST(ScalarLog, Test2) {
    auto s1 = Scalar<float>::make(2.0);
    auto s2 = log(s1);

    EXPECT_FLOAT_EQ(s2->value, log(2.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, 0.5);
}

TEST(ScalarLog, Test3) {
    auto s1 = Scalar<float>::make(10.0);
    auto s2 = log(s1);

    EXPECT_FLOAT_EQ(s2->value, log(10.0));

    s2->backward();
    EXPECT_FLOAT_EQ(s1->grad, 0.1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}