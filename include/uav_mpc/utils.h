#include <tf/tf.h>
#include <iostream>
#include <Eigen/Core>

#include <uav_mpc/types.h>


#include <distance_map_msgs/DistanceMap.h>
#include <distance_map_core/distance_map.h>


namespace utils{
    inline distmap::DistanceMap distmap_from_msg(const distance_map_msgs::DistanceMap &msg)
    {
        distmap::DistanceMap grid(distmap::DistanceMap::Dimension(msg.info.width, msg.info.height),
                                msg.info.resolution,
                                distmap::DistanceMap::Origin(msg.info.origin.position.x,
                                                            msg.info.origin.position.y,
                                                            tf::getYaw(msg.info.origin.orientation)));

        std::copy(msg.data.data(),
                msg.data.data() + (msg.info.width * msg.info.height),
                grid.data());

        return grid;
    }

    inline double compute_arclen(const std::vector<SplineWrapper> &traj, double t0, double tf)
    {
        double s = 0.0;
        double dt = (tf - t0) / 10.;
        for (double t = t0; t <= tf; t += dt)
        {
            double dx = traj[0].spline.deriv(1, t);
            double dy = traj[1].spline.deriv(1, t);

            s += std::sqrt(dx * dx + dy * dy) * dt;
        }
        return s;
    }

    inline double binary_search(const std::vector<SplineWrapper> &traj, double dl, double start, double end, double tolerance)
    {
        double t_left = start;
        double t_right = end;

        double prev_s = 0;
        double s = -1000;

        while (fabs(prev_s - s) > tolerance)
        {
            prev_s = s;

            double t_mid = (t_left + t_right) / 2;
            s = compute_arclen(traj, start, t_mid);

            if (s < dl)
                t_left = t_mid;
            else
                t_right = t_mid;
        }

        return (t_left + t_right) / 2;
    }

}
