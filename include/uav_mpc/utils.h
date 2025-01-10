#include <tf/tf.h>
#include <iostream>
#include <Eigen/Core>

#include <uav_mpc/types.h>

#include <grid_map_ros/grid_map_ros.hpp>

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

    inline bool raycast_grid(const Eigen::Vector2d& start, 
                             const Eigen::Vector2d& dir, 
                             const grid_map::GridMap& grid_map,
                             double max_dist,
                             double& actual_dist)
    {
        grid_map::Position end = start + max_dist * dir;

        // get indices in map
        grid_map::Index start_ind;
        if (!grid_map.getIndex(start, start_ind))
            return false;

        grid_map::Index end_ind;
        if (!grid_map.getIndex(end, end_ind))
            return false;

        Eigen::Vector2d ray_end = end;
        for(grid_map::LineIterator iterator(grid_map, start_ind, end_ind);
            !iterator.isPastEnd(); ++iterator)
        {
            if(grid_map.at("layer", *iterator) == 100)
            {

                // I'm pretty sure this isn't possible
                if (!grid_map.getPosition(*iterator, ray_end))
                    return false;

                break;
            }
        }

        actual_dist = (start - ray_end).norm();
        return true;
    }

    inline bool get_tubes(const std::vector<SplineWrapper> &traj, 
                          double traj_arc_len,
                          double len_start,
                          double horizon,
                          const grid_map::GridMap& grid_map,
                          std::vector<SplineWrapper>& tubes)
    {
        tubes.clear();
        std::vector<double> ss, ds_above, ds_below;

        // double horizon = 1;
        traj_arc_len -= .1;
        if (len_start > traj_arc_len)
            return false;

        if (len_start + horizon > traj_arc_len)
            horizon = traj_arc_len-len_start;

        // iterate over curve to find maximum distance allowed
        double max_dist = .7;
        double max_curv = -1;
        double max_curv_s = 0;
        double tan_mag_s = 0;
        for(double s = len_start; s <= len_start + horizon; s += .05)
        {
            double px = traj[0].spline(s);
            double py = traj[1].spline(s);
            
            double tx = traj[0].spline.deriv(1, s);
            double ty = traj[1].spline.deriv(1, s);

            double nx = traj[0].spline.deriv(2, s);
            double ny = traj[1].spline.deriv(2, s);

            Eigen::Vector2d point(px, py);
            Eigen::Vector2d normal(nx, ny);

            double den = tx * tx + ty * ty;
            double curvature = fabs(tx*ny - ty*nx) / (den * sqrt(den)); //normal.norm();

            if (curvature > 1 / (2*max_dist))
            {
                max_dist = 1 / (2*curvature);
            }

            if (curvature > max_curv)
            {
                max_curv = curvature;
                max_curv_s = s;
                tan_mag_s = Eigen::Vector2d(tx, ty).norm();
            }

        }
        
        std::cout << "MAX CURV IS: " << max_curv << " AT " << max_curv_s << " / " << traj_arc_len << std::endl;
        std::cout << "TAN MAG IS: " << tan_mag_s << std::endl;
        std::cout << "MAX DIST IS: " << max_dist << std::endl;
        for(double s = len_start; s <= len_start + horizon; s += .05)
        {
            double px = traj[0].spline(s);
            double py = traj[1].spline(s);

            double tx = traj[0].spline.deriv(1, s);
            double ty = traj[1].spline.deriv(1, s);

            Eigen::Vector2d point(px, py);
            Eigen::Vector2d normal(-ty, tx);
            normal.normalize();

            // now raycast in each normal direction
            double dist_above;
            if (!raycast_grid(point, normal, grid_map, max_dist, dist_above))
                return false;

            double dist_below;
            if (!raycast_grid(point, -1*normal, grid_map, max_dist, dist_below))
                return false;

            // std::cout << "above for " << point.transpose() << " is " << dist_above << std::endl;
            // std::cout << "below for " << point.transpose() << " is " << dist_below<< std::endl;
            
            ss.push_back(s);
            ds_above.push_back(dist_above);
            ds_below.push_back(dist_below);
        }

        double thresh = .05;
        // Backward smoothing pass
        for (int i = 0; i < ds_above.size()-1; ++i)
        {
            if (ds_above[i + 1] - ds_above[i] > thresh) // Threshold for large jump
            {
                ds_above[i + 1] = ds_above[i] + thresh/4;
            }
            if (ds_below[i + 1] - ds_below[i] > thresh)
            {
                ds_below[i + 1] = ds_below[i] + thresh/4;
            }
        }

        // Forward smoothing pass
        for (int i = ds_above.size()-2; i >= 0; --i)
        {
            if (ds_above[i] - ds_above[i + 1] > thresh)
            {
                ds_above[i] = ds_above[i + 1] + thresh/4;
            }
            if (ds_below[i] - ds_below[i + 1] > thresh)
            {
                ds_below[i] = ds_below[i + 1] + thresh/4;
            }
        }


        tk::spline spline_above(ss, ds_above, tk::spline::cspline);
        tk::spline spline_below(ss, ds_below, tk::spline::cspline);

        SplineWrapper sw_above, sw_below;
        sw_above.spline = spline_above;
        sw_below.spline = spline_below;

        tubes.push_back(sw_above);
        tubes.push_back(sw_below);

        return true;
    }
}
