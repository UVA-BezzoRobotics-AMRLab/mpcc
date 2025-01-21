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
	//states whether the trajectory is currently being modified / blended between and old and new trajectory
	_in_transition = false;
    _transition_duration = 1.0;  
    _old_ref_len = 0;

	_estop = false;
	_is_init = false;
	_is_goal = false;
	_traj_reset = false;
	_is_executing = false;

	_curr_vel = 0;
	_ref_len = 0;
	_requested_len = 0;
	_ref = {};
	_requested_ref = {};
	_requested_ss = {};
	_requested_xs = {};
	_requested_ys = {};

	_dist_grid_ptr =
		std::make_shared<distmap::DistanceMap>(distmap::DistanceMap::Dimension(5, 5),
											   1,
											   distmap::DistanceMap::Origin());
	auto _x_goal_euclid = 0;
	auto _y_goal_euclid = 0;

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

	_odomSub = nh.subscribe("/odometry/filtered", 1, &MPCCROS::odomcb, this);
	_trajSub = nh.subscribe("/reference_trajectory", 1, &MPCCROS::trajectorycb, this);
	_distMapSub = nh.subscribe("/distance_map_node/distance_field_obstacles", 1, &MPCCROS::distmapcb, this);

	_timer = nh.createTimer(ros::Duration(_dt), &MPCCROS::controlLoop, this);
	// _velPubTimer = nh.createTimer(ros::Duration(1./_vel_pub_freq), &MPCCROS::publishVel, this);

	_pathPub = nh.advertise<nav_msgs::Path>("/reference_path", 10);
	_velPub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
	_trajPub = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
	_solveTimePub = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
	_pointPub = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
	_goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
	_tubeVizPub = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
	_horizonPub = nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
	_refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/current_reference", 10);

	_modify_traj_srv = nh.advertiseService("/modify_trajectory", &MPCCROS::modifyTrajSrv, this);
	_execute_traj_srv = nh.advertiseService("/execute_trajectory", &MPCCROS::executeTrajSrv, this);

	timer_thread = std::thread(&MPCCROS::publishVel, this);
}

MPCCROS::~MPCCROS()
{
	if (timer_thread.joinable())
		timer_thread.join();

	// delete _mpc;
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

	 ROS_DEBUG("Received odometry: x=%.2f, y=%.2f", msg->pose.pose.position.x, msg->pose.pose.position.y);

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

	if (msg->points.empty()) {
        	ROS_WARN("Received empty trajectory, ignoring.");
        	return;
    	}

    	ROS_INFO("Received new trajectory with %zu points.", msg->points.size());


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

	if (_ref_len <= 0) {
        	ROS_WARN("Reference trajectory length is non-positive. Check your trajectory input!");
    	} else {
        	ROS_INFO("Set reference trajectory. ref_len=%.2f", _ref_len);
    	}
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

	if (!_is_executing) {
        return;
    }

	if (_ref.size() != 0)
	{
		// generate tubes
		// std::vector<SplineWrapper> tubes;
		// bool status = utils::get_tubes(_ref, _ref_len, _grid_map, tubes);

		std::vector<double> mpc_results = _mpc_core->solve();

		ros::Time begin = ros::Time::now();

		velMsg.linear.x = mpc_results[0];
		velMsg.angular.z = mpc_results[1];

		publishMPCTrajectory();
	}
}

bool MPCCROS::modifyTrajSrv(uvatraj_msgs::ExecuteTraj::Request &req, 
					   uvatraj_msgs::ExecuteTraj::Response &res)
{

	// dummy time variable to parameterize initial spline
	double T = 1.0;
	double dt = T / req.ctrl_pts.size();

	std::vector<double> tt, xt, yt;

	// create initial spline out of the points being sent
	double t = 0;
	for(int i = 0; i < req.ctrl_pts.size(); ++i)
	{
		tt.push_back(t);
		xt.push_back(req.ctrl_pts[i].x);
		yt.push_back(req.ctrl_pts[i].y);
		t += dt;
	}

    tk::spline spline_x(tt, xt, tk::spline::cspline);
    tk::spline spline_y(tt, yt, tk::spline::cspline);

	std::vector<SplineWrapper> ref;
	SplineWrapper sx_wrap;
    sx_wrap.spline = spline_x;

    SplineWrapper sy_wrap;
    sy_wrap.spline = spline_y;

	ref.push_back(sx_wrap);
	ref.push_back(sy_wrap);

	// get points of equal arc length along the trajectory...
	std::vector<double> ss, xs, ys;

	double M = 20;
	ss.resize(M + 1);
    xs.resize(M + 1);
    ys.resize(M + 1);

	double total_length = utils::compute_arclen(ref, 0, 1);
	double ds = total_length / M;

    for (int i = 0; i <= M; ++i)
    {
        double s = i * ds;

        double ti = utils::binary_search(ref, s, 0, 1, 1e-3);

        ss[i] = s;
		xs[i] = ref[0].spline(ti);
		ys[i] = ref[1].spline(ti);
    }
	   
    _requested_ref.clear();
	_requested_len = ss.back();

	tk::spline refx_spline(ss, xs, tk::spline::cspline);
	tk::spline refy_spline(ss, ys, tk::spline::cspline);

    SplineWrapper refx;
    refx.spline = refx_spline;

    SplineWrapper refy;
    refy.spline = refy_spline;

	_requested_ref.push_back(refx);
	_requested_ref.push_back(refy);

	_requested_ss = ss;
	_requested_xs = xs;
	_requested_ys = ys;
 	if (_is_executing)
    {
        // Store old trajectory for blending
        _old_ref = _ref;
        _old_ref_len = _ref_len;
        
        // Store new trajectory
        _ref = _requested_ref;
        _ref_len = _requested_len;
        
        // Initialize transition
        _transition_start_time = ros::Time::now().toSec();
        _transition_duration = 1.0; 
        _in_transition = true;
        
        
        _mpc_core->set_trajectory(_requested_ss, _requested_xs, _requested_ys);
        ROS_INFO("Trajectory modification started - transitioning smoothly.");
    }
    else
    {
        ROS_INFO("Trajectory modified but not applied. Call executeTrajSrv to start.");
    }

	publishReference(_requested_ref, _requested_len);

	ROS_ERROR("SPLINE RECEIVED OK!!!!!!!");
	ROS_ERROR("******************REFERENCE GENERATED******************");

	return true;
}

//claude.ai

void MPCCROS::blendTrajectories(double blend_factor)
{
    // Use normalized coordinates from 0 to 1
    double num_points = 10;
    for (double t = 0; t <= 1.0; t += 1.0/num_points)
    {
        // Scale s coordinate for each trajectory
        double s_old = t * _old_ref_len;
        double s_new = t * _ref_len;
        
        // Get points from both trajectories
        double old_x = _old_ref[0].spline(s_old);
        double old_y = _old_ref[1].spline(s_old);
        double new_x = _ref[0].spline(s_new);
        double new_y = _ref[1].spline(s_new);
        
        // Linear interpolation between trajectories
        double blended_x = old_x * (1 - blend_factor) + new_x * blend_factor;
        double blended_y = old_y * (1 - blend_factor) + new_y * blend_factor;
        
        // Scale s coordinate for the blended trajectory
        double s_blended = t * std::min(_old_ref_len, _ref_len);
        _mpc_core->updateReferencePoint(s_blended, blended_x, blended_y);
    }
}

bool MPCCROS::executeTrajSrv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{

	ROS_INFO("Execute trajectory service called.");
	    if (!_is_init) {
	        ROS_WARN("Cannot execute trajectory: tracker not initialized (no odom?).");
	        return false;
	    }
	
	    if (_is_executing) {
	        ROS_WARN("Already executing a trajectory. No changes made.");
	        return false;
	    }
	
	    if (_requested_ref.empty()) {
	        ROS_WARN("No requested reference stored. Call modifyTrajSrv first or provide a trajectory.");
	        return false;
	    }
	
	    if (_requested_len <= 0) {
	        ROS_WARN("Requested reference length <= 0. Trajectory is invalid.");
	        return false;
	    }

	if (!_is_executing){
		_ref = _requested_ref;
		_ref_len = _requested_len;
		_traj_reset = true;
		_is_executing = true; 
		_mpc_core->set_trajectory(_requested_ss, _requested_xs, _requested_ys);
		_x_goal_euclid = _requested_xs.back(); 
        _y_goal_euclid = _requested_ys.back();  
		ROS_INFO("Executing the requested trajectory now. Robot will start moving.");
		return true;
    
	} 
	else {
		ROS_WARN("Already executing a trajectory. No changes made.");
		return false;
	}
	

	return true;
}

void MPCCROS::controlLoop(const ros::TimerEvent &)
{

	ROS_DEBUG("Entered control loop.");
	
	// if not initialized
    if (!_is_init) {
        ROS_DEBUG("Not initialized yet. Waiting for odom...");
        return;
    }
	//if 
    if (_in_transition) {
        double current_time = ros::Time::now().toSec();
        double elapsed_time = current_time - _transition_start_time;
        ROS_DEBUG("In transition: elapsed=%.2f/%.2f", elapsed_time, _transition_duration);

        if (elapsed_time >= _transition_duration) {
            _in_transition = false;
            ROS_INFO("Transition complete.");
        } else {
            double blend_factor = 0.5 * (1 - cos(M_PI * elapsed_time / _transition_duration));
            ROS_DEBUG("Blend factor=%.3f", blend_factor);
            blendTrajectories(blend_factor);
        }
    }
	//calcu euclid distance
	double dx = _odom(0) - _x_goal_euclid;  //distance x
    double dy = _odom(1) - _y_goal_euclid;	 // distance y
    double dist_to_goal = std::sqrt(dx * dx + dy * dy);

    // If within some threshold
	ROS_WARN("Outside loop (dist_to_goal: %.2f) (_y_goal: %.2f) (_x_goal: %.2f)", dist_to_goal, _y_goal_euclid, _x_goal_euclid);
    if (dist_to_goal < 0.1) {
        ROS_WARN("Close enough to goal (%.2f < %.2f). Stopping execution.", dist_to_goal, _tol);

        _is_executing = false;
    }

	//bugs it out so that it send cmd_vel commands that are essentially 0, 2e^-16
    /*if (_is_executing) {
        double progress = _mpc_core->getTrajectoryProgress();
        ROS_WARN("Trajectory progress: %.2f%%", progress * 100.0);
        if (progress >= 0.99 && progress >= 0.1 && !std::isnan(progress) ) {  
            _is_executing = false;
            _in_transition = false;
            _traj_reset = false;

			velMsg.linear.x = 0.0;
        	velMsg.angular.z = 0.0;
			velMsg.angular.y = 0.0;
			
            ROS_WARN("Trajectory execution complete. Stopping.");
			return;
        }
    }*/

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
			publishReference(_ref, _ref_len);

			velMsg.linear.x = 0;
			velMsg.angular.z = std::max(-_max_angvel, std::min(_max_angvel, _prop_gain * e));
			return;
		}
	}

	cte_ctrl_loop();
}

void MPCCROS::publishReference(const std::vector<SplineWrapper>& ref, double ref_len)
{

	if (ref.size() == 0)
		return;

	nav_msgs::Path msg;
	msg.header.stamp = ros::Time::now();
	msg.header.frame_id = _frame_id;

	bool published = false;
	for (double s = 0; s < ref_len; s+=.05)
	{
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = ros::Time::now();
		pose.header.frame_id = _frame_id;

		pose.pose.position.x = ref[0].spline(s);
		pose.pose.position.y = ref[1].spline(s);
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
