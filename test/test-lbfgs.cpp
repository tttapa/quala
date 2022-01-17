#include <quala/lbfgs.hpp>

#include <limits>

#include "eigen-matchers.hpp"
#include <Eigen/LU>

TEST(LBFGS, quadratic) {
    quala::mat H(2, 2);
    H << 2, -1, -1, 3;

    std::cout << "Inverse Hessian: \n" << H.inverse() << std::endl;

    auto f      = [&H](quala::crvec v) { return 0.5 * v.dot(H * v); };
    auto grad_f = [&H](quala::crvec v) { return quala::vec(H * v); };

    quala::mat B = quala::mat::Identity(2, 2);

    quala::LBFGSParams param;
    param.memory = 5;
    quala::LBFGS lbfgs(param, 2);
    quala::vec x(2);
    x << 10, -5;
    auto r                = grad_f(x);
    unsigned update_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        { // Print L-BFGS inverse Hessian estimate
            std::cout << std::endl << i << std::endl;
            std::cout << "x:    " << x.transpose() << std::endl;
            std::cout << "f(x): " << f(x) << std::endl;

            quala::mat H⁻¹ = quala::mat::Identity(2, 2);
            if (i > 0) {
                lbfgs.apply(H⁻¹.col(0), 1);
                lbfgs.apply(H⁻¹.col(1), 1);
            }
            std::cout << std::endl << "LB⁻¹ = \n" << H⁻¹ << std::endl;
            std::cout << "B⁻¹  = \n" << B.inverse() << std::endl;
            std::cout << "   " << update_count << std::endl;
        }

        quala::vec d = r;
        if (i > 0)
            lbfgs.apply(d, 1);
        quala::vec x_new = x - d;
        quala::vec r_new = grad_f(x_new);
        lbfgs.update(x, x_new, r, r_new, quala::LBFGS::Sign::Positive);
        ++update_count;

        quala::vec y = r_new - r;
        quala::vec s = -d;
        B            = B + y * y.transpose() / y.dot(s) -
            (B * s) * (s.transpose() * B.transpose()) / (s.transpose() * B * s);

        r = std::move(r_new);
        x = std::move(x_new);
    }
    std::cout << std::endl << "final" << std::endl;
    std::cout << "x:    " << x.transpose() << std::endl;
    std::cout << "f(x): " << f(x) << std::endl;

    EXPECT_NEAR(x(0), 0, 1e-10);
    EXPECT_NEAR(x(1), 0, 1e-10);
}

TEST(BFGS, matrix) {
    quala::mat A(2, 2);
    A << 20, -10, -10, 30;

    std::cout << "A: \n" << A << std::endl;
    std::cout << "A⁻¹: \n" << A.inverse() << std::endl;
    quala::mat Bₖ = quala::mat::Identity(2, 2);

    quala::vec xₖ(2), b(2);
    b << 10, 20;
    xₖ = -b;
    quala::vec rₖ = A * xₖ - b;
    std::vector<quala::real_t> res;
    unsigned update_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        { // Print BFGS estimate
            std::cout << "\nIter: " << i << std::endl;
            std::cout << "x:    " << xₖ.transpose() << std::endl;
            std::cout << "Ax - b: " << (A * xₖ - b).transpose() << std::endl;
            std::cout << "updates: " << update_count << std::endl;
            std::cout << "B = \n" << Bₖ << std::endl;
            std::cout << "B⁻¹ = \n" << Bₖ.inverse() << std::endl;
            res.push_back((A * xₖ - b).norm());
        }

        quala::vec dₖ = -Bₖ * rₖ;
        dₖ = -Bₖ.partialPivLu().solve(rₖ);
        quala::vec xₖ₊₁ = xₖ + dₖ;
        quala::vec rₖ₊₁ = A * xₖ₊₁ - b;

        quala::vec yₖ = rₖ₊₁ - rₖ;
        quala::vec sₖ = xₖ₊₁ - xₖ;
        quala::mat Bₖ₊₁(2, 2);

        if (false) { // BFGS
            Bₖ₊₁ = Bₖ - (Bₖ * sₖ) * (Bₖ * sₖ).transpose() / sₖ.dot(Bₖ * sₖ) +
                   yₖ * yₖ.transpose() / yₖ.dot(sₖ);
        } else if (false) { // BFGS inverse
            quala::real_t ρ = 1 / sₖ.dot(yₖ);
            auto I          = quala::mat::Identity(2, 2);
            Bₖ₊₁            = (I - ρ * sₖ * yₖ.transpose()) * Bₖ *
                       (I - ρ * yₖ * sₖ.transpose()) +
                   ρ * sₖ * sₖ.transpose();
        } else if (true) { // Broyden
            Bₖ₊₁ = Bₖ + ((yₖ - Bₖ * sₖ) * sₖ.transpose()) / sₖ.squaredNorm();
        } else { // Broyden inverse
            Bₖ₊₁ = Bₖ + ((sₖ - Bₖ * yₖ) / (sₖ.transpose() * Bₖ * yₖ)) *
                            sₖ.transpose() * Bₖ;
        }
        ++update_count;

        rₖ = std::move(rₖ₊₁);
        xₖ = std::move(xₖ₊₁);
        Bₖ = std::move(Bₖ₊₁);
    }
    std::cout << "\nfinal" << std::endl;
    std::cout << "x:    " << xₖ.transpose() << std::endl;
    std::cout << "Ax - b: " << (A * xₖ - b).transpose() << std::endl;
    std::cout << "[";
    for (auto r : res)
        std::cout << r << ", ";
    std::cout << "]" << std::endl;

    EXPECT_NEAR(xₖ(0), 1, 1e-10);
    EXPECT_NEAR(xₖ(1), 1, 1e-10);
}
