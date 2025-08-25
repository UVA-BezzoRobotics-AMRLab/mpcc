#include "mpcc/mpcc_ros.h"
#include <nav_msgs/Path.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <algorithm>
#include <boost/tuple/detail/tuple_basic.hpp>

#include "mpcc/utils.h"

MPCCROS::MPCCROS(ros::NodeHandle& nh) : _nh("~")
{
    _estop        = false;
    _is_init      = false;
    _is_goal      = false;
    _traj_reset   = false;
    _reverse_mode = false;

    _curr_vel     = 0;
    _ref_len      = 0;
    _prev_ref_len = 0;
    _true_ref_len = 0;

    _s_dot  = 0;
    _prev_s = 0;

    _ref = {};

    _vel_msg.linear.x  = 0;
    _vel_msg.angular.z = 0;

    double freq;

    // Localization params
    _nh.param("use_vicon", _use_vicon, false);

    // MPC params
    _nh.param("vel_pub_freq", _vel_pub_freq, 20.0);
    _nh.param("controller_frequency", freq, 10.0);
    _nh.param("mpc_steps", _mpc_steps, 10.0);
    _nh.param("cmd_acc", _cmd_acc, false);

    // Cost function params
    _nh.param("w_vel", _w_vel, 1.0);
    _nh.param("w_angvel", _w_angvel, 1.0);
    _nh.param("w_linvel", _w_linvel, 1.0);
    _nh.param("w_angvel_d", _w_angvel_d, 1.0);
    _nh.param("w_linvel_d", _w_linvel_d, .5);
    _nh.param("w_etheta", _w_etheta, 1.0);
    _nh.param("w_cte", _w_cte, 1.0);

    _nh.param("w_lag_e", _w_ql, 50.0);
    _nh.param("w_contour_e", _w_qc, .1);
    _nh.param("w_speed", _w_q_speed, .3);

    // Constraint params
    _nh.param("w_max", _max_angvel, 3.0);
    _nh.param("v_max", _max_linvel, 2.0);
    _nh.param("a_max", _max_linacc, 3.0);
    _nh.param("anga_max", _max_anga, 2 * M_PI);
    _nh.param("bound_value", _bound_value, 1.0e19);

    // Goal params
    _nh.param("x_goal", _x_goal, 0.0);
    _nh.param("y_goal", _y_goal, 0.0);
    _nh.param("goal_tolerance", _tol, 0.3);

    // Teleop params
    _nh.param("teleop", _teleop, false);
    _nh.param<std::string>("frame_id", _frame_id, "map");

    // clf params
    _nh.param("w_lyap_lag_e", _w_ql_lyap, 1.0);
    _nh.param("w_lyap_contour_e", _w_qc_lyap, 1.0);
    _nh.param("clf_gamma", _clf_gamma, .5);

    // cbf params
    _nh.param("use_cbf", _use_cbf, false);
    _nh.param("cbf_alpha_abv", _cbf_alpha_abv, .5);
    _nh.param("cbf_alpha_blw", _cbf_alpha_blw, .5);
    _nh.param("cbf_colinear", _cbf_colinear, .1);
    _nh.param("cbf_padding", _cbf_padding, .1);
    _nh.param("dynamic_alpha", _use_dynamic_alpha, false);

    // proportional controller params
    _nh.param("prop_gain", _prop_gain, .5);
    _nh.param("prop_gain_thresh", _prop_angle_thresh, 30. * M_PI / 180.);

    // tube parameters
    _nh.param("tube_poly_degree", _tube_degree, 6);
    _nh.param("tube_num_samples", _tube_samples, 50);
    _nh.param("max_tube_width", _max_tube_width, 2.0);

    _nh.param("ref_length_size", _mpc_ref_len_sz, 4.);
    _nh.param("mpc_ref_samples", _mpc_ref_samples, 10);

    _nh.param("/train/logging", _is_logging, false);
    _nh.param("/train/is_eval", _is_eval, false);
    _nh.param("/train/min_alpha", _min_alpha, .1);
    _nh.param("/train/max_alpha", _max_alpha, 10.);

    // num coeffs is tube_W_ANGVELdegree + 1
    _tube_degree += 1;

    // tube width technically from traj to tube boundary
    _max_tube_width /= 2;

    _dt  = 1.0 / freq;
    _cmd = Command(_cmd_acc);

    _mpc_params["DT"]      = _dt;
    _mpc_params["STEPS"]   = _mpc_steps;
    _mpc_params["CMD_ACC"] = _cmd_acc;

    _mpc_params["W_V"]       = _w_linvel;
    _mpc_params["W_ANGVEL"]  = _w_angvel;
    _mpc_params["W_DA"]      = _w_linvel_d;
    _mpc_params["W_DANGVEL"] = _w_angvel_d;
    _mpc_params["W_ETHETA"]  = _w_etheta;
    _mpc_params["W_POS"]     = _w_pos;
    _mpc_params["W_CTE"]     = _w_cte;
    _mpc_params["LINVEL"]    = _max_linvel;
    _mpc_params["ANGVEL"]    = _max_angvel;
    _mpc_params["BOUND"]     = _bound_value;
    _mpc_params["X_GOAL"]    = _x_goal;
    _mpc_params["Y_GOAL"]    = _y_goal;

    _mpc_params["ANGLE_THRESH"] = _prop_angle_thresh;
    _mpc_params["ANGLE_GAIN"]   = _prop_gain;

    _mpc_params["W_LAG"]     = _w_ql;
    _mpc_params["W_CONTOUR"] = _w_qc;
    _mpc_params["W_SPEED"]   = _w_q_speed;

    _mpc_params["REF_LENGTH"]  = _mpc_ref_len_sz;
    _mpc_params["REF_SAMPLES"] = _mpc_ref_samples;

    _mpc_params["CLF_GAMMA"]     = _clf_gamma;
    _mpc_params["CLF_W_LAG"]     = _w_ql_lyap;
    _mpc_params["CLF_W_CONTOUR"] = _w_qc_lyap;

    _mpc_params["USE_CBF"]           = _use_cbf;
    _mpc_params["CBF_ALPHA_ABV"]     = _cbf_alpha_abv;
    _mpc_params["CBF_ALPHA_BLW"]     = _cbf_alpha_blw;
    _mpc_params["CBF_COLINEAR"]      = _cbf_colinear;
    _mpc_params["CBF_PADDING"]       = _cbf_padding;
    _mpc_params["CBF_DYNAMIC_ALPHA"] = _use_dynamic_alpha;

    _mpc_params["MAX_ANGA"]   = _max_anga;
    _mpc_params["MAX_LINACC"] = _max_linacc;

    _mpc_params["DEBUG"] = true;

    _mpc_core = std::make_unique<MPCCore>();
    ROS_INFO("loading mpc params");
    _mpc_core->load_params(_mpc_params);
    ROS_INFO("done loading params!");

    _mapSub  = nh.subscribe("/map", 1, &MPCCROS::mapcb, this);
    _odomSub = nh.subscribe("/ROSBOT08/odom", 1, &MPCCROS::odomcb, this);
    _trajSub = nh.subscribe("/TransferTrajectory", 1, &MPCCROS::trajectorycb, this);
    _obsSub  = nh.subscribe("/obs_odom", 1, &MPCCROS::dynaobscb, this);

    _timer = nh.createTimer(ros::Duration(_dt), &MPCCROS::mpcc_ctrl_loop, this);
    // _velPubTimer = nh.createTimer(ros::Duration(1./_vel_pub_freq),
    // &MPCCROS::publishVel, this);

    _startPub       = nh.advertise<std_msgs::Float64>("/progress", 10);
    _pathPub        = nh.advertise<nav_msgs::Path>("/spline_path", 10);
    _velPub         = nh.advertise<geometry_msgs::Twist>("/ROSBOT08/cmd_vel", 10);
    _trajPub        = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
    _solveTimePub   = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
    _goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
    _pointPub       = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
    _refVizPub      = nh.advertise<visualization_msgs::Marker>("/mpc_reference", 0);
    _tubeVizPub     = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
    _horizonPub     = nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
    _refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/current_reference", 10);

    timer_thread = std::thread(&MPCCROS::publishVel, this);

    _backup_srv = nh.advertiseService("/mpc_backup", &MPCCROS::toggleBackup, this);
    _execute_traj_srv = nh.advertiseService("/execute_trajectory", &MPCCROS::executeTrajSrv, this);


    _is_executing = false;


    if (_is_logging)
    {
        ROS_WARN("******************");
        ROS_WARN("LOGGING IS ENABLED");
        ROS_WARN("******************");
    }

    if (_is_logging || _is_eval)
        _logger = std::make_unique<logger::RLLogger>(nh, _min_alpha, _max_alpha, _is_logging);
}

MPCCROS::~MPCCROS()
{
    if (timer_thread.joinable()) timer_thread.join();
}



bool MPCCROS::executeTrajSrv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res){

	ROS_INFO("Execute trajectory service called.");

	//if (!_is_init) {
	        //ROS_WARN("Cannot execute trajectory: tracker not initialized (no odom?).");
	      //  return false;
	    //}

	    //if (_is_executing) {
	       // ROS_WARN("Already executing a trajectory. No changes made.");
	       // return false;
	    ///}

	//if (!_is_executing){
		_traj_reset = true;
		_is_executing = true;
		_mpc_core->set_trajectory(_ref, _ref_len, _true_ref_len);
		ROS_INFO("Executing the requested trajectory now. Robot will start moving.");
		return true;



	//}
	//else {
		//ROS_WARN("Already executing a trajectory. No changes made.");
		//return true;
	//}


	return true;


} 


bool MPCCROS::toggleBackup(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
    _reverse_mode = !_reverse_mode;
    return true;
}

void MPCCROS::visualizeTubes()
{
    const Eigen::VectorXd& state = _mpc_core->get_state();
    double len_start             = state(4);
    double horizon               = _max_linvel * _dt * _mpc_steps;

    if (len_start > _true_ref_len) return;

    if (len_start + horizon > _true_ref_len) horizon = _true_ref_len - len_start;

    Eigen::VectorXd abv_coeffs = _tubes[0];
    Eigen::VectorXd blw_coeffs = _tubes[1];

    visualization_msgs::Marker tubemsg_a;
    tubemsg_a.header.frame_id    = _frame_id;
    tubemsg_a.header.stamp       = ros::Time::now();
    tubemsg_a.ns                 = "tube_above";
    tubemsg_a.id                 = 87;
    tubemsg_a.action             = visualization_msgs::Marker::ADD;
    tubemsg_a.type               = visualization_msgs::Marker::LINE_STRIP;
    tubemsg_a.scale.x            = .075;
    tubemsg_a.pose.orientation.w = 1;

    visualization_msgs::Marker tubemsg_b = tubemsg_a;
    tubemsg_b.header                     = tubemsg_a.header;
    tubemsg_b.ns                         = "tube_below";
    tubemsg_b.id                         = 88;

    visualization_msgs::Marker normals_msg;
    normals_msg.header.frame_id = _frame_id;
    normals_msg.ns              = "traj_normals";
    normals_msg.id              = 237;
    normals_msg.action          = visualization_msgs::Marker::ADD;
    normals_msg.type            = visualization_msgs::Marker::LINE_LIST;
    normals_msg.scale.x         = .01;

    normals_msg.color.r = 1.0;
    normals_msg.color.a = 1.0;

    visualization_msgs::Marker normals_below_msg = normals_msg;
    normals_below_msg.ns                         = "traj_normals_below";
    normals_below_msg.id                         = 238;
    normals_below_msg.color.r                    = 0.0;
    normals_below_msg.color.b                    = 1.0;

    // if horizon is that small, too small to visualize anyway
    if (horizon < .05) return;

    tubemsg_a.points.reserve(2 * (horizon / .05));
    tubemsg_b.points.reserve(2 * (horizon / .05));
    tubemsg_a.colors.reserve(2 * (horizon / .05));
    tubemsg_b.colors.reserve(2 * (horizon / .05));
    normals_msg.points.reserve(2 * (horizon / .05));
    normals_below_msg.points.reserve(2 * (horizon / .05));
    for (double s = len_start; s < len_start + horizon; s += .05)
    {
        double px = _ref[0](s).coeff(0);
        double py = _ref[1](s).coeff(0);

        double tx = _ref[0].derivatives(s, 1).coeff(1);
        double ty = _ref[1].derivatives(s, 1).coeff(1);

        Eigen::Vector2d point(px, py);
        Eigen::Vector2d normal(-ty, tx);
        normal.normalize();

        double da = utils::eval_traj(abv_coeffs, s - len_start);
        double db = utils::eval_traj(blw_coeffs, s - len_start);

        geometry_msgs::Point& pt_a = tubemsg_a.points.emplace_back();
        pt_a.x                     = point(0) + normal(0) * da;
        pt_a.y                     = point(1) + normal(1) * da;
        pt_a.z                     = 1.0;

        geometry_msgs::Point& pt_b = tubemsg_b.points.emplace_back();
        pt_b.x                     = point(0) + normal(0) * db;
        pt_b.y                     = point(1) + normal(1) * db;
        pt_b.z                     = 1.0;

        // if (fabs(da) > 1.1 || fabs(db) > 1.1)
        // {
        //     ROS_WARN("s: %.2f\tda = %.2f, db = %.2f", s, da, db);
        //     should_exit = true;
        // }

        // convenience for setting colors
        std_msgs::ColorRGBA color_msg_abv;
        color_msg_abv.r = 192. / 255.;
        color_msg_abv.g = 0.0;
        color_msg_abv.b = 0.0;
        color_msg_abv.a = 1.0;

        std_msgs::ColorRGBA color_msg_blw;
        color_msg_blw.r = 251. / 255.;
        color_msg_blw.g = 133. / 255.;
        color_msg_blw.b = 0.0;
        color_msg_blw.a = 1.0;

        tubemsg_a.colors.push_back(color_msg_abv);
        tubemsg_b.colors.push_back(color_msg_blw);

        geometry_msgs::Point normal_pt;
        normal_pt.x = point(0);
        normal_pt.y = point(1);
        normal_pt.z = 1.0;

        // make normals msg instead show tangent line
        Eigen::Vector2d tangent(tx, ty);
        tangent.normalize();
        geometry_msgs::Point tangent_pt;
        tangent_pt.x = point(0) + .025 * tangent(0);
        tangent_pt.y = point(1) + .025 * tangent(1);
        tangent_pt.z = 1.0;

        normals_msg.points.push_back(normal_pt);
        normals_msg.points.push_back(tangent_pt);

        // normals_msg.points.push_back(normal_pt);
        // normals_msg.points.push_back(pt_a);

        // normals_below_msg.points.push_back(normal_pt);
        // normals_below_msg.points.push_back(pt_b);
    }

    visualization_msgs::MarkerArray tube_ma;
    tube_ma.markers.reserve(2);
    tube_ma.markers.push_back(std::move(tubemsg_a));
    tube_ma.markers.push_back(std::move(tubemsg_b));
    // tube_ma.markers.push_back(std::move(normals_msg));
    // tube_ma.markers.push_back(std::move(normals_below_msg));

    _tubeVizPub.publish(tube_ma);

    // if (should_exit)
    // {
    //     ROS_WARN("exiting due to large tube values");
    //     exit(1);
    // }
}

void MPCCROS::visualizeTraj()
{
    visualization_msgs::Marker traj;
    traj.header.frame_id    = _frame_id;
    traj.header.stamp       = ros::Time::now();
    traj.ns                 = "mpc_reference";
    traj.id                 = 117;
    traj.action             = visualization_msgs::Marker::ADD;
    traj.type               = visualization_msgs::Marker::LINE_STRIP;
    traj.scale.x            = .075;
    traj.pose.orientation.w = 1;

    for (double s = 0; s < _true_ref_len; s += .05)
    {
        double px = _ref[0](s).coeff(0);
        double py = _ref[1](s).coeff(0);

        geometry_msgs::Point& pt_a = traj.points.emplace_back();
        pt_a.x                     = px;
        pt_a.y                     = py;
        pt_a.z                     = 0.5;

        std_msgs::ColorRGBA color_msg;
        color_msg.r = 1.0;
        color_msg.g = 140. / 255.;
        color_msg.b = 0.0;
        color_msg.a = 1.0;

        traj.colors.push_back(color_msg);
    }

    _refVizPub.publish(traj);
}

void MPCCROS::publishVel()
{
    constexpr double pub_vel_loop_rate_hz = 50;
    const std::chrono::milliseconds pub_loop_period(
        static_cast<int>(1000.0 / pub_vel_loop_rate_hz));

    while (ros::ok())
    {
        /*if (_trajectory.points.size() > 0) _velPub.publish(_vel_msg);*/
        if (_is_executing) _velPub.publish(_cmd.get_cmd());

        // _velPub.publish(_vel_msg);

        std::this_thread::sleep_for(pub_loop_period);
    }
}

void MPCCROS::mapcb(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "layer", _grid_map);
}

void MPCCROS::goalcb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    _x_goal = msg->pose.position.x;
    _y_goal = msg->pose.position.y;

    _is_goal    = true;
    _is_at_goal = false;

    ROS_WARN("GOAL RECEIVED (%.2f, %.2f)", _x_goal, _y_goal);
}

void MPCCROS::odomcb(const nav_msgs::Odometry::ConstPtr& msg)
{
    tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    _odom = Eigen::VectorXd(3);

    _odom(0) = msg->pose.pose.position.x;
    _odom(1) = msg->pose.pose.position.y;
    _odom(2) = yaw;

    if (_reverse_mode)
    {
        _odom(2) += M_PI;
        // wrap to pi
        if (_odom(2) > M_PI) _odom(2) -= 2 * M_PI;
    }

    _mpc_core->set_odom(_odom);

    if (!_is_init)
    {
        _is_init = true;
        ROS_INFO("tracker initialized");
    }
}

void MPCCROS::dynaobscb(const nav_msgs::Odometry::ConstPtr& msg)
{
    Eigen::MatrixXd dyna_obs(3, 2);
    dyna_obs.col(0) << msg->pose.pose.position.x, msg->pose.pose.position.y,
        msg->pose.pose.position.z;
    dyna_obs.col(1) << msg->twist.twist.linear.x, msg->twist.twist.linear.y,
        msg->pose.pose.position.z;

    _mpc_core->set_dyna_obs(dyna_obs);
}

void recalculateArcLength(const Eigen::RowVectorXd& xs,
                          const Eigen::RowVectorXd& ys,
                          Eigen::RowVectorXd& ss)
{
    const int n_pts = xs.size();
    ss.resize(n_pts);
    if (n_pts == 0) {
        return;
    }

    ss(0) = 0.0;
    for (int i = 1; i < n_pts; ++i) {
        double dx = xs(i) - xs(i - 1);
        double dy = ys(i) - ys(i - 1);
        ss(i) = ss(i - 1) + std::hypot(dx, dy);
    }
}



/**********************************************************************
 * Function: MPCCROS::trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr& msg)
 * Description: Callback for trajectory message
 * Parameters:
 * @param msg: trajectory_msgs::JointTrajectory::ConstPtr
 * Returns:
 * N/A
 * Notes:
 * This function sets the reference trajectory for the MPC controller
 * Since the ACADOS MPC requires a hard coded trajectory size, the
 * trajectory is extended if it is less than the required size
 **********************************************************************/
void MPCCROS::trajectorycb(const nav_msgs::Path::ConstPtr& msg)
{
    ROS_INFO("Trajectory received!");
    //_trajectory = *msg;

    if (msg->poses.size() == 0)
    {
        ROS_WARN("Trajectory is empty, stopping!");
        /*_vel_msg.linear.x  = std::max(_vel_msg.linear.x - _max_linacc * _dt, 0.0);*/
        /*_vel_msg.angular.z = 0;*/

        if (_cmd_acc)
        {
            double old_vel    = _cmd.vel.linear.x;
            _cmd.vel.linear.x = std::max(_cmd.vel.linear.x - _max_linacc * _dt, 0.0);
            _cmd.acc.linear.x = (_cmd.vel.linear.x - old_vel) / _dt;
        }
        else
        {
            _cmd.vel.linear.x = std::max(_vel_msg.linear.x - _max_linacc * _dt, 0.0);
            _cmd.acc.linear.x = 0;
        }

        return;
    }

    _prev_ref     = _ref;
    _prev_ref_len = _true_ref_len;

    _traj_reset = true;

    int N = msg->poses.size();

    Eigen::RowVectorXd ss, xs, ys;
    ss.resize(N);
    xs.resize(N);
    ys.resize(N);

    for (int i = 0; i < N; ++i)
    {
        xs(i) = msg->poses[i].pose.position.x;
        ys(i) = msg->poses[i].pose.position.y;
        //ss(i) = msg->points[i].time_from_start.toSec();

        ROS_INFO("added pt (%.2f, %.2f)", xs(i), ys(i));
    }

    recalculateArcLength(xs,ys,ss);

    _ref_len      = ss(ss.size() - 1);
    _true_ref_len = _ref_len;

    const auto fitX = utils::Interp(xs, 3, ss);
    Spline1D splineX(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    Spline1D splineY(fitY);

    // if reference length is less than required mpc size, extend trajectory
    if (_ref_len < _mpc_ref_len_sz)
    {
        ROS_WARN("reference length (%.2f) is smaller than %.2fm, extending", _ref_len,
                 _mpc_ref_len_sz);

        double end = _ref_len - 1e-1;
        double px  = splineX(end).coeff(0);
        double py  = splineY(end).coeff(0);
        double dx  = splineX.derivatives(end, 1).coeff(1);
        double dy  = splineY.derivatives(end, 1).coeff(1);

        /*ROS_WARN("(%.2f, %.2f)\t(%.2f, %.2f)", px, py, dx, dy);*/

        double ds = _mpc_ref_len_sz / (N - 1);

        for (int i = 0; i < N; ++i)
        {
            double s = ds * i;
            ss(i)    = s;

            if (s < _ref_len)
            {
                xs(i) = splineX(s).coeff(0);
                ys(i) = splineY(s).coeff(0);
            }
            else
            {
                xs(i) = dx * (s - _ref_len) + px;
                ys(i) = dy * (s - _ref_len) + py;
            }
        }

        const auto fitX = utils::Interp(xs, 3, ss);
        splineX         = Spline1D(fitX);

        const auto fitY = utils::Interp(ys, 3, ss);
        splineY         = Spline1D(fitY);

        _ref_len = _mpc_ref_len_sz;
    }

    _ref[0] = splineX;
    _ref[1] = splineY;

    _mpc_core->set_trajectory(_ref, _ref_len, _true_ref_len);

    visualizeTraj();

    ROS_INFO("**********************************************************");
    ROS_INFO("MPC received trajectory! Length: %.2f", _ref_len);
    ROS_INFO("**********************************************************");
}

double MPCCROS::get_s_from_state(const std::array<Spline1D, 2>& ref, double ref_len)
{
    // find the s which minimizes dist to robot
    double s            = 0;
    double min_dist     = 1e6;
    Eigen::Vector2d pos = _odom.head(2);
    for (double i = 0.0; i < ref_len; i += .01)
    {
        Eigen::Vector2d p = Eigen::Vector2d(ref[0](i).coeff(0), ref[1](i).coeff(0));

        double d = (pos - p).squaredNorm();
        if (d < min_dist)
        {
            min_dist = d;
            s        = i;
        }
    }

    return s;
}

void MPCCROS::mpcc_ctrl_loop(const ros::TimerEvent& event)
{
    if (!_is_init || _estop) return;

    //if (_trajectory.points.size() == 0) return;
    
    if (!_is_executing) return;


    double len_start     = get_s_from_state(_ref, _true_ref_len);
    double corrected_len = len_start;

    ROS_INFO("len_start is: %.2f / %.2f", len_start, _true_ref_len);

    std_msgs::Float64 start_msg;
    start_msg.data = len_start / _true_ref_len;
    _startPub.publish(start_msg);

    // correct len_start and _prev_s if trajectory reset
    if (!_traj_reset) _s_dot = std::min(std::max((len_start - _prev_s) / _dt, 0.), _max_linvel);

    if (_traj_reset && _prev_ref_len > 0)
    {
        // get arc_len of previous trajectory
        /*double s = get_s_from_state(_prev_ref, _prev_ref_len);*/
        /*corrected_len += s;*/
        /*len_start = get_s_from_state(_prev_ref, _prev_ref_len);*/

        _traj_reset = false;
    }

    /*_s_dot = std::min(std::max((corrected_len - _prev_s) / _dt, 0.), _max_linvel);*/
    /*_prev_s = len_start;*/

    _prev_s = get_s_from_state(_ref, _true_ref_len);

    /*if (len_start > _true_ref_len - 3e-1)*/
    /*{*/
    /*    ROS_INFO("Reached end of traj %.2f / %.2f", len_start, _true_ref_len);*/
    /*    // apply maximum deceleration until robot stops*/
    /*    if (_cmd_acc)*/
    /*    {*/
    /*        double old_vel = _cmd.vel.linear.x;*/
    /*        _cmd.vel.linear.x = std::max(_cmd.vel.linear.x - _max_linacc * _dt, 0.0);*/
    /*        _cmd.acc.linear.x = (_cmd.vel.linear.x - old_vel) / _dt;*/
    /**/
    /*        if (fabs(_cmd.vel.linear.x) < 1e-2)*/
    /*          _trajectory.points.clear();*/
    /*    }*/
    /*    else*/
    /*    {*/
    /*        _cmd.vel.linear.x = std::max(_vel_msg.linear.x - _max_linacc * _dt, 0.0);*/
    /*        _cmd.acc.linear.x = 0;*/
    /*        _trajectory.points.clear();*/
    /*    }*/
    /**/
    /*    return;*/
    /*}*/

    double horizon = _mpc_ref_len_sz;

    if (len_start + horizon > _ref_len) horizon = _ref_len - len_start;

    // generate tubes
    // std::vector<SplineWrapper> tubes;
    ros::Time now = ros::Time::now();
    bool status   = true;
    if (_use_cbf)
    {
        std::cout << "ref_len size is: " << _ref_len << std::endl;
        status = utils::get_tubes(_tube_degree, _tube_samples, _max_tube_width, _ref, _ref_len,
                                  len_start, horizon, _odom, _grid_map, _tubes);

        ROS_INFO("finished tube generation");
    }
    else
    {
        Eigen::VectorXd upper_coeffs(_tube_degree);
        Eigen::VectorXd lower_coeffs(_tube_degree);

        upper_coeffs.setZero();
        lower_coeffs.setZero();
        upper_coeffs(0) = 100;
        lower_coeffs(0) = -100;

        _tubes[0] = upper_coeffs;
        _tubes[1] = lower_coeffs;
    }

    if (!status)
        ROS_WARN("could not generate tubes, mpc not running");
    else
        //visualizeTubes();
    ROS_WARN("set tubes");
    _mpc_core->set_tubes(_tubes);

    // get alpha value if logging is enabled
    if (_is_logging || _is_eval)
    {
        // request alpha sets the alpha
        bool status = _logger->request_alpha(*_mpc_core, _true_ref_len);
        if (!status)
        {
            ROS_WARN("could not get alpha value from logger");
            return;
        }
    }

    Eigen::VectorXd state(6);
    double vel = _cmd.vel.linear.x;
    if (_reverse_mode) vel *= -1;
    state << _odom(0), _odom(1), _odom(2), vel, 0, _s_dot;


    ROS_WARN("mpc_results");
    std::array<double, 2> mpc_results = _mpc_core->solve(state, _reverse_mode);
    ROS_WARN("after mpc_results");
    _cmd.vel.angular.z = mpc_results[1];
    _cmd.acc.angular.z = mpc_results[1];
    if (!_cmd_acc)
        _cmd.vel.linear.x = mpc_results[0];
    else
    {
        _cmd.vel.linear.x = _mpc_core->get_vel();
        _cmd.acc.linear.x = mpc_results[0];

        ROS_INFO("CMD ACC IS %.2f", _cmd.acc.linear.x);
        ROS_INFO("CMD OMEGA IS %.2f", _cmd.acc.angular.z);
    }

    // log data back to db if logging enabled
    if (_is_logging || _is_eval) _logger->log_transition(*_mpc_core, len_start, _true_ref_len);

    publishReference();
    publishMPCTrajectory();

    geometry_msgs::PointStamped pt;
    pt.header.frame_id = _frame_id;
    pt.point.z         = .1;

    double s = get_s_from_state(_ref, _true_ref_len);

    pt.header.stamp = ros::Time::now();
    pt.point.x      = _ref[0](s).coeff(0);
    pt.point.y      = _ref[1](s).coeff(0);

    _pointPub.publish(pt);
}

void MPCCROS::publishReference()
{
    if (_trajectory.points.size() == 0) return;

    nav_msgs::Path msg;
    msg.header.stamp    = ros::Time::now();
    msg.header.frame_id = _frame_id;
    msg.poses.reserve(_trajectory.points.size());

    bool published = false;
    for (const trajectory_msgs::JointTrajectoryPoint& pt : _trajectory.points)
    {
        if (!published)
        {
            published = true;
            _refPub.publish(pt);
        }

        geometry_msgs::PoseStamped& pose = msg.poses.emplace_back();
        pose.header.stamp                = ros::Time::now();
        pose.header.frame_id             = _frame_id;
        pose.pose.position.x             = pt.positions[0];
        pose.pose.position.y             = pt.positions[1];
        pose.pose.position.z             = 0;
        pose.pose.orientation.x          = 0;
        pose.pose.orientation.y          = 0;
        pose.pose.orientation.z          = 0;
        pose.pose.orientation.w          = 1;
    }

    _pathPub.publish(msg);
}

void MPCCROS::publishMPCTrajectory()
{
    std::vector<Eigen::VectorXd> horizon = _mpc_core->get_horizon();

    if (horizon.size() == 0) return;

    geometry_msgs::PoseStamped goal;
    goal.header.stamp       = ros::Time::now();
    goal.header.frame_id    = _frame_id;
    goal.pose.position.x    = _x_goal;
    goal.pose.position.y    = _y_goal;
    goal.pose.orientation.w = 1;

    nav_msgs::Path pathMsg;
    pathMsg.header.frame_id = _frame_id;
    pathMsg.header.stamp    = ros::Time::now();

    for (int i = 0; i < horizon.size(); ++i)
    {
        // don't visualize mpc horizon past end of reference trajectory
        if (horizon[i](6) > _true_ref_len) break;

        Eigen::VectorXd state = horizon[i];
        geometry_msgs::PoseStamped tmp;
        tmp.header             = pathMsg.header;
        tmp.pose.position.x    = state(1);
        tmp.pose.position.y    = state(2);
        tmp.pose.position.z    = 1.0;
        tmp.pose.orientation.w = 1;
        pathMsg.poses.push_back(tmp);
    }

    _trajPub.publish(pathMsg);

    if (horizon.size() > 1)
    {
        // convert to JointTrajectory
        trajectory_msgs::JointTrajectory traj;
        traj.header.stamp    = ros::Time::now();
        traj.header.frame_id = _frame_id;

        double dt = horizon[1](0) - horizon[0](0);

        for (int i = 0; i < horizon.size(); ++i)
        {
            Eigen::VectorXd state = horizon[i];

            double t      = state(0);
            double x      = state(1);
            double y      = state(2);
            double theta  = state(3);
            double linvel = state(4);
            double linacc = state(5);

            // compute velocity and acceleration in x and y directions
            double vel_x = linvel * cos(theta);
            double vel_y = linvel * sin(theta);

            // ROS_INFO("vel_x and vel_y: %.2f, %.2f", vel_x, vel_y);

            double acc_x = linacc * cos(theta);
            double acc_y = linacc * sin(theta);
            // ROS_INFO("acc_x and acc_y: %.2f, %.2f", acc_x, acc_y);

            // manually compute jerk in x and y directions from acceleration
            double jerk_x = 0;
            double jerk_y = 0;
            if (i < horizon.size() - 1)
            {
                double next_linacc   = horizon[i + 1](5);
                double next_linacc_x = next_linacc * cos(horizon[i + 1](3));
                double next_linacc_y = next_linacc * sin(horizon[i + 1](3));
                jerk_x               = (next_linacc_x - acc_x) / dt;
                jerk_y               = (next_linacc_y - acc_y) / dt;

                // ROS_INFO("jerk_x = %.2f, jerk_y = %.2f", jerk_x,
                // jerk_y);
            }
            else
            {
                jerk_x = 0;
                jerk_y = 0;

                // ROS_INFO("(in else cond) jerk_x = 0, jerk_y =
                // 0");
            }

            trajectory_msgs::JointTrajectoryPoint pt;
            pt.time_from_start = ros::Duration(t);
            pt.positions       = {x, y, 0};
            pt.velocities      = {vel_x, vel_y, 0};
            pt.accelerations   = {acc_x, acc_y, 0};
            pt.effort          = {jerk_x, jerk_y, 0};

            traj.points.push_back(pt);
        }

        _horizonPub.publish(traj);
    }
}
