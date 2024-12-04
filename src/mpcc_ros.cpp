#include <math.h>
#include <algorithm>
#include <ros/ros.h>

#include <tf/tf.h>
#include <Eigen/Core>
#include <std_msgs/Bool.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PolygonStamped.h>

#include "uav_mpc/mpcc_ros.h"
#include "uav_mpc/utils.h"

MPCCROS::MPCCROS(ros::NodeHandle &nh) : _nh("~")
{

	_estop = false;
	_is_init = false;
	_is_goal = false;
	_traj_reset = false;

	_curr_vel = 0;
	_ref_len = 0;
	_ref = {};

	_dist_grid_ptr =
		std::make_shared<distmap::DistanceMap>(distmap::DistanceMap::Dimension(5, 5),
											   1,
											   distmap::DistanceMap::Origin());


	velMsg.linear.x = 0;
	velMsg.angular.z = 0;

	double freq;

	// Localization params
	_nh.param("use_vicon", _use_vicon, false);

	// MPC params
	_nh.param("vel_pub_freq", _vel_pub_freq, 20.0);
	_nh.param("controller_frequency", freq, 10.0);
	_nh.param("mpc_steps", _mpc_steps, 10.0);

	// Cost function params
	_nh.param("w_vel", _w_vel, 1.0);
	_nh.param("w_angvel", _w_angvel, 1.0);
	_nh.param("w_linvel", _w_linvel, 1.0);
	_nh.param("w_angvel_d", _w_angvel_d, 1.0);
	_nh.param("w_linvel_d", _w_linvel_d, .5);
	_nh.param("w_etheta", _w_etheta, 1.0);
	_nh.param("w_cte", _w_cte, 1.0);
	_nh.param("w_pos", _w_pos, 1.0);

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
	_nh.param<std::string>("frame_id", _frame_id, "odom");

	// cbf params
	_nh.param("use_cbf", _use_cbf, false);
	_nh.param("cbf_alpha", _cbf_alpha, .5);
	_nh.param("cbf_colinear", _cbf_colinear, .1);
	_nh.param("cbf_padding", _cbf_padding, .1);
	_nh.param("dynamic_alpha", _use_dynamic_alpha, false);

	// proportional controller params
	_nh.param("prop_gain", _prop_gain, .5);
	_nh.param("prop_angle_thresh", _prop_angle_thresh, 30. * M_PI / 180.);

	_dt = 1.0 / freq;

	_mpc_params["DT"] = _dt;
	_mpc_params["STEPS"] = _mpc_steps;
	_mpc_params["W_V"] = _w_linvel;
	_mpc_params["W_ANGVEL"] = _w_angvel;
	_mpc_params["W_DA"] = _w_linvel_d;
	_mpc_params["W_DANGVEL"] = _w_angvel_d;
	_mpc_params["W_ETHETA"] = _w_etheta;
	_mpc_params["W_POS"] = _w_pos;
	_mpc_params["W_CTE"] = _w_cte;
	_mpc_params["LINVEL"] = _max_linvel;
	_mpc_params["ANGVEL"] = _max_angvel;
	_mpc_params["BOUND"] = _bound_value;
	_mpc_params["X_GOAL"] = _x_goal;
	_mpc_params["Y_GOAL"] = _y_goal;

	_mpc_params["USE_CBF"] = _use_cbf;
	_mpc_params["CBF_ALPHA"] = _cbf_alpha;
	_mpc_params["CBF_COLINEAR"] = _cbf_colinear;
	_mpc_params["CBF_PADDING"] = _cbf_padding;
	_mpc_params["CBF_DYNAMIC_ALPHA"] = _use_dynamic_alpha;

	_mpc_params["MAX_ANGA"] = _max_anga;
	_mpc_params["MAX_LINACC"] = _max_linacc;

	_mpc_params["DEBUG"] = true;

	_mpc_core = std::make_unique<MPCCore>();
	ROS_INFO("loading mpc params");
	_mpc_core->load_params(_mpc_params);
	ROS_INFO("done loading params!");

	_mapSub = nh.subscribe("/map", 1, &MPCCROS::mapcb, this);
	_odomSub = nh.subscribe("/odometry/filtered", 1, &MPCCROS::odomcb, this);
	_trajSub = nh.subscribe("/reference_trajectory", 1, &MPCCROS::trajectorycb, this);
	_distMapSub = nh.subscribe("/distance_map_node/distance_field_obstacles", 1, &MPCCROS::distmapcb, this);

	_timer = nh.createTimer(ros::Duration(_dt), &MPCCROS::controlLoop, this);
	// _velPubTimer = nh.createTimer(ros::Duration(1./_vel_pub_freq), &MPCCROS::publishVel, this);

	_pathPub = nh.advertise<nav_msgs::Path>("/spline_path", 10);
	_velPub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
	_trajPub = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
	_solveTimePub = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
	_pointPub = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
	_goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
	_tubeVizPub = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
	_horizonPub = nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
	_refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/current_reference", 10);

	timer_thread = std::thread(&MPCCROS::publishVel, this);
}

MPCCROS::~MPCCROS()
{
	if (timer_thread.joinable())
		timer_thread.join();

	// delete _mpc;
}

void MPCCROS::visualizeTubes()
{

	Eigen::VectorXd state = _mpc_core->get_state();
	double len_start = state(4);
	double horizon = 1.0;

	if (len_start > _ref_len)
		return;

	if (len_start + horizon > _ref_len)
		horizon = _ref_len - len_start;

	std::vector<SplineWrapper> tubes;
	if(!utils::get_tubes(_ref, _ref_len, len_start, _grid_map, tubes))
		return;

	tk::spline d_above = tubes[0].spline;
	tk::spline d_below = tubes[1].spline;

	visualization_msgs::Marker tubemsg_a;
	tubemsg_a.header.frame_id = _frame_id;
	tubemsg_a.header.stamp = ros::Time::now();
	tubemsg_a.ns = "tube_above";
	tubemsg_a.id = 87;
	tubemsg_a.action = visualization_msgs::Marker::ADD;
	tubemsg_a.type = visualization_msgs::Marker::LINE_STRIP;
	tubemsg_a.scale.x = .05;
	tubemsg_a.pose.orientation.w = 1;

	visualization_msgs::Marker tubemsg_b = tubemsg_a;
	tubemsg_b.ns = "tube_below";
	tubemsg_b.id = 88;

	for (double s = len_start; s < len_start + 1; s += .05)
	{

		// get point and tangent to curve
		double px = _ref[0].spline(s);
		double py = _ref[1].spline(s);

		double tx = _ref[0].spline.deriv(1, s);
		double ty = _ref[1].spline.deriv(1, s);

		std_msgs::ColorRGBA color_msg;
		color_msg.r = 0.0;
		color_msg.g = 1.0;
		color_msg.b = 1.0;
		color_msg.a = 1.0;

		Eigen::Vector2d point(px, py);
		Eigen::Vector2d normal(-ty, tx);
		normal = normal / normal.norm();

		double da = d_above(s);
		double db = d_below(s);

		geometry_msgs::Point tube_pt;
		tube_pt.x = point(0) + normal(0) * da;
		tube_pt.y = point(1) + normal(1) * da;
		tube_pt.z = 1.0;
		tubemsg_a.points.push_back(tube_pt);

		geometry_msgs::Point tube_pt1;
		tube_pt1.x = point(0) - normal(0) * db;
		tube_pt1.y = point(1) - normal(1) * db;
		tube_pt1.z = 1.0;
		tubemsg_b.points.push_back(tube_pt1);

		tubemsg_a.colors.push_back(color_msg);
		tubemsg_b.colors.push_back(color_msg);
	}

	visualization_msgs::MarkerArray tube_ma;
	tube_ma.markers.push_back(tubemsg_a);
	tube_ma.markers.push_back(tubemsg_b);

	_tubeVizPub.publish(tube_ma);
}


void MPCCROS::publishVel()
{
	constexpr double pub_vel_loop_rate_hz = 50;
	const std::chrono::milliseconds pub_loop_period(static_cast<int>(1000.0 / pub_vel_loop_rate_hz));

	while (ros::ok())
	{
		if (trajectory.points.size() > 0)
		{
			_refPub.publish(current_reference);
		}

		// ROS_INFO("PUBLISHING {vel = %.2f, ang_z = %.2f}", velMsg.linear.x, velMsg.angular.z);
		_velPub.publish(velMsg);
		std::this_thread::sleep_for(pub_loop_period);
	}
}

void MPCCROS::alphacb(const std_msgs::Float64::ConstPtr &msg)
{
	_mpc_params["CBF_ALPHA"] = msg->data;
	_mpc_core->load_params(_mpc_params);
}

void MPCCROS::mapcb(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
	grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "layer", _grid_map);
}

void MPCCROS::goalcb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
	_x_goal = msg->pose.position.x;
	_y_goal = msg->pose.position.y;

	_is_goal = true;
	_is_at_goal = false;

	ROS_WARN("GOAL RECEIVED (%.2f, %.2f)", _x_goal, _y_goal);
}

void MPCCROS::odomcb(const nav_msgs::Odometry::ConstPtr &msg)
{

	tf::Quaternion q(
		msg->pose.pose.orientation.x,
		msg->pose.pose.orientation.y,
		msg->pose.pose.orientation.z,
		msg->pose.pose.orientation.w);

	tf::Matrix3x3 m(q);
	double roll, pitch, yaw;
	m.getRPY(roll, pitch, yaw);

	_odom = Eigen::VectorXd(3);

	_odom(XI) = msg->pose.pose.position.x;
	_odom(YI) = msg->pose.pose.position.y;
	_odom(THETAI) = yaw;

	_mpc_core->set_odom(_odom);

	if (!_is_init)
	{
		_is_init = true;
		ROS_INFO("tracker initialized");
	}
}

// TODO: Support appending trajectories
void MPCCROS::trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr &msg)
{
	trajectory = *msg;
	_traj_reset = true;

	int N = msg->points.size();
	// Eigen::RowVectorXd ss, xs, ys;
	// ss.resize(N);
	// xs.resize(N);
	// ys.resize(N);

	// for (int i = 0; i < N; ++i)
	// {
	// 	xs(i) = msg->points[i].positions[0];
	// 	ys(i) = msg->points[i].positions[1];
	// 	ss(i) = msg->points[i].time_from_start.toSec();
	// }
	std::vector<double> ss, xs, ys;
	ss.resize(N);
	xs.resize(N);
	ys.resize(N);

	for (int i = 0; i < N; ++i)
	{
		xs[i] = msg->points[i].positions[0];
		ys[i] = msg->points[i].positions[1];
		ss[i] = msg->points[i].time_from_start.toSec();

	}

	for (int i = 0; i < N-1; ++i)
	{
		if (ss[i] >= ss[i + 1]){
			std::cerr << "NOT MONOTONIC IN S!!" << std::endl;
			std::cerr << i << ": " << ss[i] << " --> " << i+1 << ": " << ss[i+1] << std::endl;
		}
	}

    // tk::spline sx(ss, xs, tk::spline::cspline, false,
    //               tk::spline::first_deriv, 1.0,
    //               tk::spline::first_deriv, 1.0);
    // tk::spline sy(ss, ys, tk::spline::cspline, false,
    //               tk::spline::first_deriv, 1.0,
    //               tk::spline::first_deriv, 1.0);

    tk::spline sx(ss, xs, tk::spline::cspline);
    tk::spline sy(ss, ys, tk::spline::cspline);

    _ref.clear();
	_ref_len = ss.back();

    SplineWrapper sx_wrap;
    sx_wrap.spline = sx;

    SplineWrapper sy_wrap;
    sy_wrap.spline = sy;

	_ref.push_back(sx_wrap);
	_ref.push_back(sy_wrap);

	_mpc_core->set_trajectory(ss, xs, ys);

	ROS_INFO("**********************************************************");
	ROS_INFO("MPC received trajectory!");
	ROS_INFO("**********************************************************");
}

void MPCCROS::distmapcb(const distance_map_msgs::DistanceMap::ConstPtr &msg)
{
	if (msg == nullptr)
	{
		ROS_WARN("Distance map is null");
		return;
	}

	*_dist_grid_ptr = utils::distmap_from_msg(*msg);
	_mpc_core->set_dist_map(_dist_grid_ptr);
}

void MPCCROS::cte_ctrl_loop()
{
	static ros::Time start;

	if (!_is_init || _estop)
		return;

	if (_traj_reset)
	{
		start = ros::Time::now();
		_traj_reset = false;
	}

	if (trajectory.points.size() != 0)
	{
		// generate tubes
		// std::vector<SplineWrapper> tubes;
		// bool status = utils::get_tubes(_ref, _ref_len, _grid_map, tubes);

		std::vector<double> mpc_results = _mpc_core->solve();

		ros::Time begin = ros::Time::now();
		visualizeTubes();
		ROS_INFO("tube generation time is: %.4f", (ros::Time::now() - begin).toSec());

		velMsg.linear.x = mpc_results[0];
		velMsg.angular.z = mpc_results[1];

		publishReference();
		publishMPCTrajectory();
	}
}

void MPCCROS::controlLoop(const ros::TimerEvent &)
{
	// don't care about aligning if trajectory short
	if (_ref_len > 1 && _traj_reset)
	{
		// calculate heading error between robot and trajectory start
		// use 1st point as most times first point has 0 velocity

		double traj_heading = atan2(_ref[1].spline.deriv(1, .2),
									_ref[0].spline.deriv(1, .2));

		// wrap between -pi and pi
		double e = atan2(sin(traj_heading - _odom(THETAI)), cos(traj_heading - _odom(THETAI)));

		ROS_INFO("Robot heading is %.2f and traj_heading is %.2f", _odom(THETAI), traj_heading);
		ROS_WARN("trajectory reset, checking if we need to align... error = %.2f deg", e * 180. / M_PI);

		// if error is larger than _prop_angle_thresh use proportional controller to align
		if (fabs(e) > _prop_angle_thresh)
		{
			ROS_WARN("Alignment to trajectory now!");
			trajectory_msgs::JointTrajectory traj;
			traj.header.stamp = ros::Time::now();
			traj.header.frame_id = _frame_id;
			for (int i = 0; i < _mpc_steps; ++i)
			{
				trajectory_msgs::JointTrajectoryPoint pt;
				pt.positions = {_odom(XI), _odom(YI), 0};
				pt.velocities = {0, 0, 0};
				pt.accelerations = {0, 0, 0};
				pt.effort = {0, 0, 0};
				pt.time_from_start = ros::Duration(i * _dt);
				traj.points.push_back(pt);
			}

			_horizonPub.publish(traj);
			publishReference();

			velMsg.linear.x = 0;
			velMsg.angular.z = std::max(-_max_angvel, std::min(_max_angvel, _prop_gain * e));
			return;
		}
	}

	cte_ctrl_loop();
}

void MPCCROS::publishReference()
{

	if (trajectory.points.size() == 0)
		return;

	nav_msgs::Path msg;
	msg.header.stamp = ros::Time::now();
	msg.header.frame_id = _frame_id;

	bool published = false;
	for (trajectory_msgs::JointTrajectoryPoint pt : trajectory.points)
	{
		if (!published)
		{
			published = true;
			_refPub.publish(pt);
		}
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = ros::Time::now();
		pose.header.frame_id = _frame_id;

		pose.pose.position.x = pt.positions[0];
		pose.pose.position.y = pt.positions[1];
		pose.pose.position.z = 0;
		pose.pose.orientation.x = 0;
		pose.pose.orientation.y = 0;
		pose.pose.orientation.z = 0;
		pose.pose.orientation.w = 1;
		msg.poses.push_back(pose);
	}

	_pathPub.publish(msg);
}

void MPCCROS::publishMPCTrajectory()
{

	std::vector<Eigen::VectorXd> horizon = _mpc_core->get_horizon();

	if (horizon.size() == 0)
		return;

	geometry_msgs::PoseStamped goal;
	goal.header.stamp = ros::Time::now();
	goal.header.frame_id = _frame_id;
	goal.pose.position.x = _x_goal;
	goal.pose.position.y = _y_goal;
	goal.pose.orientation.w = 1;

	nav_msgs::Path pathMsg;
	pathMsg.header.frame_id = _frame_id;
	pathMsg.header.stamp = ros::Time::now();

	for (int i = 0; i < horizon.size(); ++i)
	{
		Eigen::VectorXd state = horizon[i];
		geometry_msgs::PoseStamped tmp;
		tmp.header = pathMsg.header;
		if (state.size() == 6)
		{
			tmp.pose.position.x = state(1);
			tmp.pose.position.y = state(2);
		}
		else
		{
			tmp.pose.position.x = state(0);
			tmp.pose.position.y = state(1);
		}
		tmp.pose.position.z = .1;
		tmp.pose.orientation.w = 1;
		pathMsg.poses.push_back(tmp);
	}

	_trajPub.publish(pathMsg);

	if (horizon.size() > 1 && horizon[0].size() == 6)
	{
		// convert to JointTrajectory
		trajectory_msgs::JointTrajectory traj;
		traj.header.stamp = ros::Time::now();
		traj.header.frame_id = _frame_id;

		double dt = horizon[1](0) - horizon[0](0);

		for (int i = 0; i < horizon.size(); ++i)
		{
			Eigen::VectorXd state = horizon[i];

			double t = state(0);
			double x = state(1);
			double y = state(2);
			double theta = state(3);
			double linvel = state(4);
			double linacc = state(5);

			// compute velocity and acceleration in x and y directions
			double vel_x = linvel * cos(theta);
			double vel_y = linvel * sin(theta);

			double acc_x = linacc * cos(theta);
			double acc_y = linacc * sin(theta);

			// compute jerk in x and y directions from acceleration
			double jerk_x = 0;
			double jerk_y = 0;
			if (i < horizon.size() - 1)
			{
				double next_linacc = horizon[i + 1](5);
				double next_linacc_x = next_linacc * cos(horizon[i + 1](3));
				double next_linacc_y = next_linacc * sin(horizon[i + 1](3));
				jerk_x = (next_linacc_x - acc_x) / dt;
				jerk_y = (next_linacc_y - acc_y) / dt;

				// ROS_INFO("jerk_x = %.2f, jerk_y = %.2f", jerk_x, jerk_y);
			}
			else
			{
				jerk_x = 0;
				jerk_y = 0;

				// ROS_INFO("(in else cond) jerk_x = 0, jerk_y = 0");
			}

			trajectory_msgs::JointTrajectoryPoint pt;
			pt.time_from_start = ros::Duration(t);
			pt.positions = {x, y, 0};
			pt.velocities = {vel_x, vel_y, 0};
			pt.accelerations = {acc_x, acc_y, 0};
			pt.effort = {jerk_x, jerk_y, 0};

			traj.points.push_back(pt);
		}

		_horizonPub.publish(traj);
	}
}
