#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace bspline{

class BSpline {
public:
    BSpline(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.size() < 4) {
            throw std::invalid_argument("Input vectors must have the same size and at least 4 points.");
        }

        n = x.size();
        degree = 3; // Cubic B-spline
        knots = computeKnotVector(x);
        controlPoints = computeControlPoints(x, y);
    }

    double operator() (double t) const {
        double result = 0.0;
        for (int i = 0; i < controlPoints.size(); ++i) {
            result += controlPoints[i] * evaluateBasis(i, degree, t, knots);
        }
        return result;
    }

    const std::vector<double>& getKnots() const { return knots; }
    const std::vector<double>& getControlPoints() const { return controlPoints; }

private:
    int n;                        // Number of data points
    int degree;                   // Degree of the B-spline
    std::vector<double> knots;    // Knot vector
    std::vector<double> controlPoints; // Control points

    // Compute the knot vector with not-a-knot boundary conditions
    std::vector<double> computeKnotVector(const std::vector<double>& x) {
        std::vector<double> knotVec(n + degree + 1);
        for (int i = 0; i < knotVec.size(); ++i) {
            if (i <= degree) {
                knotVec[i] = x.front();
            } else if (i >= n) {
                knotVec[i] = x.back();
            } else {
                knotVec[i] = x[i - degree];
            }
        }
        return knotVec;
    }

    // Compute B-spline basis functions using Cox-de Boor recursion
    double evaluateBasis(int i, int k, double t, const std::vector<double>& knotVec) const {
        if (k == 1) {
            return (knotVec[i] <= t && t < knotVec[i + 1]) ? 1.0 : 0.0;
        } else {
            double denom1 = knotVec[i + k - 1] - knotVec[i];
            double denom2 = knotVec[i + k] - knotVec[i + 1];

            double coeff1 = (denom1 != 0.0) ? (t - knotVec[i]) / denom1 : 0.0;
            double coeff2 = (denom2 != 0.0) ? (knotVec[i + k] - t) / denom2 : 0.0;

            return coeff1 * evaluateBasis(i, k - 1, t, knotVec) +
                coeff2 * evaluateBasis(i + 1, k - 1, t, knotVec);
        }
    }


    // Solve for control points using interpolation
    std::vector<double> computeControlPoints(const std::vector<double>& x, const std::vector<double>& y) {
        Eigen::MatrixXd A(n, n);
        Eigen::VectorXd b(n);
        b = Eigen::Map<const Eigen::VectorXd>(y.data(), n);

        // Fill the basis matrix A
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A(i, j) = evaluateBasis(j, degree + 1, x[i], knots);
            }
        }

        // Solve for control points
        Eigen::VectorXd C = A.colPivHouseholderQr().solve(b);
        return std::vector<double>(C.data(), C.data() + C.size());
    }
};

} // end namespace bspline
