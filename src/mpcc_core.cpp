#include <chrono>
#include <cmath>
#include <iostream>

#include "uav_mpc/spline.h"
#include "uav_mpc/mpcc_core.h"

MPCCore::MPCCore()
{

    _mpc = std::make_unique<MPCC>();

    _curr_vel = 0;
    _curr_ang_vel = 0;

    _state = Eigen::VectorXd::Zero(6);

    _is_set = false;
}

MPCCore::~MPCCore()
{
}

void MPCCore::load_params(const std::map<std::string, double> &params)
{
    _dt = params.at("DT");
    _max_anga = params.at("MAX_ANGA");
    _max_linacc = params.at("MAX_LINACC");

    _max_vel = params.at("LINVEL");
    _max_ang_vel = params.at("ANGVEL");

    _mpc->load_params(params);
    // _filter.load_params(params);
}

void MPCCore::set_odom(const Eigen::Vector3d &odom)
{
    _odom = odom;
}

void MPCCore::set_trajectory(const Eigen::RowVectorXd &ss, const Eigen::RowVectorXd &xs, const Eigen::RowVectorXd &ys)
{
    _is_set = true;

    const auto fitX = SplineFitting1D::Interpolate(xs, 3, ss);
    _splineX = std::make_unique<Spline1D>(fitX);

    const auto fitY = SplineFitting1D::Interpolate(ys, 3, ss);
    _splineY = std::make_unique<Spline1D>(fitY);

    double tot_err = 0;
    for(double s = 0; s < ss(ss.size()-1); s += .01)
    {
        Eigen::Vector2d tan(_splineX->derivatives(s, 1).coeff(1), 
                            _splineY->derivatives(s, 1).coeff(1));

        tot_err += fabs(1-tan.norm());
    }

    std::cout << "EIGEN ERROR IS: " << tot_err << std::endl;

    std::vector<Spline1D> ref;
    ref.push_back(*_splineX);
    ref.push_back(*_splineY);
    _mpc->set_reference(ref, ss[ss.size() - 1]);
}

void MPCCore::set_trajectory(const std::vector<double> &ss, const std::vector<double> &xs, const std::vector<double> &ys)
{
    _is_set = true;

    // tk::spline sx(ss, xs, tk::spline::cspline, false,
    //               tk::spline::first_deriv, 1.0,
    //               tk::spline::first_deriv, 1.0);
    // tk::spline sy(ss, ys, tk::spline::cspline, false,
    //               tk::spline::first_deriv, 1.0,
    //               tk::spline::first_deriv, 1.0);

    tk::spline sx(ss, xs, tk::spline::cspline);
    tk::spline sy(ss, ys, tk::spline::cspline);

    double tot_err = 0;
    for(double s = 0; s < ss.back(); s += .01)
    {
        Eigen::Vector2d tan(sx.deriv(1, s),
                            sy.deriv(1, s));
                            

        tot_err += fabs(1-tan.norm());
    }

    std::cout << "TK ERROR IS: " << tot_err << std::endl;

    std::vector<SplineWrapper> ref;
    SplineWrapper sx_wrap;
    sx_wrap.spline = sx;

    SplineWrapper sy_wrap;
    sy_wrap.spline = sy;

    ref.push_back(sx_wrap);
    ref.push_back(sy_wrap);
    _mpc->set_reference(ref, ss[ss.size() - 1]);
}

void MPCCore::set_tubes(const std::vector<SplineWrapper>& tubes)
{
    _mpc->set_tubes(tubes);
}

double MPCCore::get_progress()
{
    double min_dist = 1e10;
    double min_time = 0;

    for (const traj_point_t &pt : _trajectory)
    {
        double dist = sqrt(
            (_odom(0) - pt.pose(0)) * (_odom(0) - pt.pose(0)) +
            (_odom(1) - pt.pose(1)) * (_odom(1) - pt.pose(1)));

        if (dist < min_dist)
        {
            min_dist = dist;
            min_time = pt.time_from_start;
        }
    }

    return min_time / _trajectory.back().time_from_start;
}

void MPCCore::set_dist_map(const std::shared_ptr<distmap::DistanceMap> &dist_map)
{
    _dist_grid_ptr = dist_map;
    // _mpc->set_dist_map(dist_map);
}

// void MPCCore::set_segments(const std::vector<Segment_t> &segments)
// {
//     _mpc->set_segments(segments);
// }

std::vector<double> MPCCore::solve()
{

    if (!_is_set)
        return {};

    // std::cout << "odometry is " << _odom.transpose() << std::endl;
    double new_vel;
    double time_to_solve = 0.;

    std::cerr << "loading state and going into mpc impl solve" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd state(4);
    state << _odom(0), _odom(1), _odom(2), _curr_vel;
    std::cerr << "done loadin state" << std::endl;
    _mpc_results = _mpc->solve(state);
    std::cerr << "done with mpc impl solve" << std::endl;
    // _mpc_results = _mpc->Solve(_state, _reference);
    auto end = std::chrono::high_resolution_clock::now();
    time_to_solve = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (time_to_solve > _dt * 1000)
    {
        _mpc_results[0] = _curr_ang_vel;
        _mpc_results[1] = 0;
    }

    new_vel = _curr_vel + _mpc_results[1] * _dt;

    // std::cerr << "mpc_results: " << _mpc_results[0] << ", " << _mpc_results[1] << std::endl;
    std::cout << "Solve time: " << time_to_solve << std::endl;

    _curr_ang_vel = limit(_curr_ang_vel, _mpc_results[0], _max_anga);
    _curr_vel = limit(_curr_vel, new_vel, _max_linacc);

    // ensure vel is between -max and max and ang vel is between -max and max
    _curr_vel = std::max(-_max_vel, std::min(_max_vel, _curr_vel));
    _curr_ang_vel = std::max(-_max_ang_vel, std::min(_max_ang_vel, _curr_ang_vel));

    std::cerr << "curr vel: " << _curr_vel << ", curr ang vel: " << _curr_ang_vel << std::endl;

    return {_curr_vel, _curr_ang_vel};
}

Eigen::VectorXd MPCCore::get_state()
{
    return _mpc->get_state();
}

std::vector<Eigen::VectorXd> MPCCore::get_horizon()
{
    std::vector<Eigen::VectorXd> ret;

    if (_mpc->mpc_x.size() == 0)
        return ret;

    double t = 0;
    for (int i = 0; i < _mpc->mpc_x.size()-1; ++i)
    {
        Eigen::VectorXd state(6);
        state << t, _mpc->mpc_x[i], _mpc->mpc_y[i], _mpc->mpc_theta[i], _mpc->mpc_linvels[i], _mpc->mpc_linaccs[i];
        t += _dt;
        ret.push_back(state);
    }

    return ret;
}

double MPCCore::limit(double prev_v, double input, double max_rate)
{
    double ret = input;
    if (fabs(prev_v - input) / _dt > max_rate)
    {

        if (input > prev_v)
            ret = prev_v + max_rate * _dt;
        else
            ret = prev_v - max_rate * _dt;
    }

    return ret;
}
