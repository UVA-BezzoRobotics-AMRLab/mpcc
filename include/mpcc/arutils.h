#pragma once

#include <vector>
#include <Eigen/Core>
#include <uvatraj_msgs/ControlPoint.h>

namespace mpcc {
namespace arutils {

/// Compute Euclidean distance between two 2D points.
double computeDistance(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b);

/// Generate a straight‐line trajectory from start → goal,
/// sampled every `resolution` (meters).
std::vector<Eigen::Vector2d> generateLinearTrajectory(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& goal,
    double                 resolution);
}  // namespace utils
}  // namespace mpcc

