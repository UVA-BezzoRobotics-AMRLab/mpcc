#pragma once

#include <uav_mpc/spline.h>
#include <unsupported/Eigen/Splines>

class SplineWrapper {
public:
    tk::spline spline; // Expose the tk::spline
};

struct traj_point
{
    Eigen::Vector3d pose;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    double time_from_start;
};
typedef struct traj_point traj_point_t;

typedef Eigen::Spline<double, 1, 3> Spline1D;
typedef Eigen::SplineFitting<Spline1D> SplineFitting1D;
