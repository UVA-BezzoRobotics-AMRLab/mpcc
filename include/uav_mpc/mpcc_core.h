#pragma once

#include <map>
#include <memory>
#include <fstream>

#include <uav_mpc/types.h>
#include <uav_mpc/mpcc_impl.h>


class MPCCore
{
public:
    MPCCore();

    ~MPCCore();

    void load_params(const std::map<std::string, double> &params);

    void set_odom(const Eigen::Vector3d &odom);
    void set_goal(const Eigen::Vector2d &goal);
    void set_trajectory(const Eigen::RowVectorXd &ss, const Eigen::RowVectorXd &xs, const Eigen::RowVectorXd &ys);
    void set_trajectory(const std::vector<double> &ss, const std::vector<double> &xs, const std::vector<double> &ys);
    void set_dist_map(const std::shared_ptr<distmap::DistanceMap> &dist_map);
    void set_tubes(const std::vector<SplineWrapper>& tubes);


    void updateReferencePoint(double s, double x, double y);

    Eigen::VectorXd get_state();
    std::vector<Eigen::VectorXd> get_horizon();

    std::vector<double> solve();

    double getTrajectoryProgress();

protected:
    double get_progress();
    double limit(double prev_v, double input, double max_rate);
    std::vector<std::tuple<double, double, double>> _blend_points;
    double _dt;
    double _max_anga;
    double _max_linacc;
    double _curr_vel;
    double _curr_ang_vel;
    double _max_vel;
    double _max_ang_vel;

    bool _use_cbf;

    std::vector<double> _mpc_results;
    std::vector<traj_point_t> _trajectory;

    Eigen::Vector3d _odom;
    Eigen::Vector2d _goal;
    Eigen::VectorXd _state;

    std::unique_ptr<Spline1D> _splineX;
    std::unique_ptr<Spline1D> _splineY;
 
	std::shared_ptr<distmap::DistanceMap> _dist_grid_ptr;

    // learning states
    Eigen::VectorXd _prev_rl_state;
    Eigen::VectorXd _curr_rl_state;

    bool _is_set;

    std::unique_ptr<MPCC> _mpc;

};
