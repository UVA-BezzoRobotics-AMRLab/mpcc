#include <vector>
#include <cmath>
#include <Eigen/Core>

#include "mpcc/arutils.h"
namespace mpcc{


	namespace arutils{
inline double computeDistance(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b)
{
    return (b - a).norm();
}

std::vector<Eigen::Vector2d> generateLinearTrajectory(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& goal,
    double                 resolution)
{
    // compute delta and length
    Eigen::Vector2d delta = goal - start;
    double         length = delta.norm();

    // handle degenerate cases
    if (length <= 0.0 || resolution <= 0.0) {
        return { start };
    }

    // unit direction
    Eigen::Vector2d dir = delta / length;

    // how many full steps of size `resolution`
    int numPts = static_cast<int>(std::floor(length / resolution));

    std::vector<Eigen::Vector2d> pts;
    pts.reserve(numPts + 2);

    // sample from i=0 (start) to i=numSteps (might land exactly on goal)
    for (int i = 0; i <= numPts; ++i) {
        pts.emplace_back(start + dir * (resolution * i));
    }

    // ensure exact goal at the end
    if ((pts.back() - goal).norm() > 1e-6) {
        pts.emplace_back(goal);
    }

    return pts;
}
}
}
