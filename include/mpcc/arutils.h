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

/// Compute the unit‐direction vector pointing from a → b.
/// Returns a zero vector if a == b.
Eigen::Vector2d computeDirection(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b);

/// Sample points along the line starting at `start` in `direction`
/// up to `total_length`, at spacing `resolution`. Includes start + goal.
std::vector<Eigen::Vector2d> sampleAlongLine(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& direction,
    double                 total_length,
    double                 resolution);

/// Generate a straight‐line trajectory from start → goal,
/// sampled every `resolution` (meters).
std::vector<Eigen::Vector2d> generateLinearTrajectory(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& goal,
    double                 resolution);

/// Convert a list of Eigen::Vector2d into ROS ControlPoint messages.
std::vector<uvatraj_msgs::ControlPoint> toControlPoints(
    const std::vector<Eigen::Vector2d>& path);

}  // namespace utils
}  // namespace mpcc

