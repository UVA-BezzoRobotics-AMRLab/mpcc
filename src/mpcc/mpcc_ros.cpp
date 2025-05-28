#include "mpcc/mpcc_ros.h"
#include <unordered_set>
#include <std_srvs/SetBool.h>
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

constexpr double resolution = 0.5;

#include "mpcc/utils.h"


//import ar utils
#include <include/mpcc/arutils.h> 
//import the uvatrajmsgs

MPCCROS::MPCCROS(ros::NodeHandle& nh) : _nh("~")
{
    _estop        = false;
    _is_init      = false;
    _is_goal      = false;
    _traj_reset   = false;
    _reverse_mode = false;
    _is_paused = false;

    _curr_vel     = 0;
    _ref_len      = 0;
    _prev_ref_len = 0;
    _true_ref_len = 0;

    _s_dot  = 0;
    _prev_s = 0;

    _ref = {};
    
    //AR
    _requested_ss = Eigen::RowVectorXd();
    _requested_xs = Eigen::RowVectorXd();
    _requested_ys = Eigen::RowVectorXd();
    _requested_ref = {};
    _old_ref = {}; 

    _blend_new_s = 0.0;
   
    _is_executing = false;
    _in_transition = false;
    _transition_start_time = 0.0;
    _transition_duration = 0.0;
    _requested_len = 0.0;
    _true_len = 0.0;
    _x_goal_euclid = 0.0;
    _y_goal_euclid = 0.0;
    _old_ref_len = 0.0;
    
    
 
    _vel_msg.linear.x  = 0;
    _vel_msg.angular.z = 0;

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
    _nh.param<std::string>("frame_id", _frame_id, "vicon/world");

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

    _dt = 1.0 / freq;

    _mpc_params["DT"]        = _dt;
    _mpc_params["STEPS"]     = _mpc_steps;
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
    //_odomSub = nh.subscribe("/odometry/filtered", 1, &MPCCROS::odomcb, this);
    _viconSub = nh.subscribe("/vicon/Rosbot_AR_2/Rosbot_AR_2", 1, &MPCCROS::viconcb, this);
    _trajSub = nh.subscribe("/reference_trajectory", 1, &MPCCROS::trajectorycb, this);
    _obsSub  = nh.subscribe("/obs_odom", 1, &MPCCROS::dynaobscb, this);

    _timer = nh.createTimer(ros::Duration(_dt), &MPCCROS::mpcc_ctrl_loop, this);
    // _velPubTimer = nh.createTimer(ros::Duration(1./_vel_pub_freq),
    // &MPCCROS::publishVel, this);

    _startPub       = nh.advertise<std_msgs::Float64>("/progress", 10);
    _pathPub        = nh.advertise<nav_msgs::Path>("/spline_path", 10);
    _velPub         = nh.advertise<geometry_msgs::Twist>("/ROSBOT8/cmd_vel", 10);
    _trajPub        = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
    _solveTimePub   = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
    _goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
    _pointPub       = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
    _refVizPub      = nh.advertise<visualization_msgs::Marker>("/mpc_reference", 0);
    _tubeVizPub     = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
    _horizonPub     = nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
    _refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/current_reference", 10);


    _generate_traj_srv = nh.advertiseService("/generate_traj", &MPCCROS::generateTrajSrv, this);
    
    _modify_traj_srv = nh.advertiseService("/modify_trajectory", &MPCCROS::modifyTrajSrv, this); 
    
    _exec_traj_Srv = nh.advertiseService("/execute_trajectory", &MPCCROS::executeTrajSrv, this);

    _pause_execution_Srv = nh.advertiseService("/pause_execution", &MPCCROS::pauseExecutionSrv, this);

    _traj_sender = nh.serviceClient<uvatraj_msgs::ExecuteTraj>("/traj_send");

    _traj_executed.clear();

    timer_thread = std::thread(&MPCCROS::publishVel, this);

    _backup_srv = nh.advertiseService("/mpc_backup", &MPCCROS::toggleBackup, this);

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

bool MPCCROS::pauseExecutionSrv(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res){


	_is_paused=!_is_paused;
	_is_executing=!_is_executing;
	res.success = true;
	res.message = "User set _is_paused to" + std::to_string(_is_paused); 
	ROS_ERROR_STREAM("USER PAUSED OR UNPAUSE"<< _is_paused);
	return true; 


}

bool MPCCROS::executeTrajSrv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res){

	ROS_INFO("Execute trajectory service called.");

	if (!_is_init) {
	        ROS_WARN("Cannot execute trajectory: tracker not initialized (no odom?).");
	        return false;
	    }
	
	    if (_is_executing) {
	        ROS_WARN("Already executing a trajectory. No changes made.");
	        return false;
	    }
	
	if (!_is_executing){
		_traj_reset = true;
		_is_executing = true;
		_in_transition = false; 
		_mpc_core->set_trajectory(_ref, _ref_len);
		ROS_INFO("Executing the requested trajectory now. Robot will start moving.");
		return true;
    
	} 
	else {
		ROS_WARN("Already executing a trajectory. No changes made.");
		return true;
	}
	

	return true;


}

//TODO: check if the modified pt is beyond where we currently are then just return true and don't do anything

/*
ool MPCCROS::modifyTrajSrv(uvatraj_msgs::ExecuteTraj::Request &req, uvatraj_msgs::ExecuteTraj::Response &res)
{


	double total_length = 0;
	int M = req.ctrl_pts.size();
	ROS_ERROR("%d POINTS RECEIVED", M);
	std::vector<double> ss,xs,ys;
	
	//add arc length at each section and x y pair at that arc length 
	for (int i = 1; i<req.ctrl_pts.size(); ++i){
	
		double x0 = req.ctrl_pts[i-1].x;
		double y0 = req.ctrl_pts[i-1].y;

		double x1 = req.ctrl_pts[i].x;
		double y1 = req.ctrl_pts[i].y;

		double dx = x1-x0;
		double dy = y1-y0;
		double segment_length = std::sqrt(dx*dx + dy*dy);
			
		if (std::fabs(segment_length) < 1e-2)
			continue;

		ss.push_back(total_length);
		xs.push_back(x0);
		ys.push_back(y0);
		total_length += segment_length;
	
	}

	//add the final pt
	ss.push_back(total_length);
	xs.push_back(req.ctrl_pts.back().x);
	ys.push_back(req.ctrl_pts.back().y);

	//construct splines parameterizing with the arc length
	tk::spline refx_spline(ss,xs,tk::spline::cspline);
	tk::spline refy_spline(ss,ys,tk::spline::cspline);

	SplineWrapper refx;
	refx.spline = refx_spline;

	SplineWrapper refy;
	refy.spline = refy_spline;

	_true_len = total_length;
	
	//DEBUG: It was stoppign before the end
	//FIX: extend the trajectory for 
	
	//get the x y pair at the end of the traj and the derivates with respect to x and y
	double px =refx.spline(total_length);
	double py =refx.spline(total_length);
	double dx = refx.spline.deriv(1,total_length);
	double dy = refx.spline.deriv(1,total_length);

	//get unit derivative/tangent vector
	double norm = std::sqrt(dx*dx + dy*dy);
	dx /= norm;
	dy /= norm;

	// could just set equal to resolution --> don't change yet
	double ds = total_length/M;
	//int num_ext_pts = 200;
	int num_ext_pts = 1;
	//extend it 200 pts???
	for (int i = 1; i<=num_ext_pts; ++i){
	double s = total_length + i*resolution;
	double x = px +dx * i * resolution;
	double y = py+dy*i*resolution;
	
	ss.push_back(s);
	xs.push_back(x);
	ys.push_back(y);
	}


	//change the spline to be the extended one / with the extended pts
	refx.spline.set_points(ss, xs, tk::spline::cspline);
	refy.spline.set_points(ss, ys, tk::spline::cspline);

	
	_requested_ref.clear();
	_requested_len = ss.back();
	
	_requested_ref.push_back(refx);
	_requested_ref.push_back(refy);
	
	_requested_ss = ss;
	_requested_xs = xs;
	_requested_ys = ys;

	if (_is_executing){
	_old_ref = _ref;
	_old_ref_len = _ref_len;
	_blend_traj_curr_s = get_s_from_state(_ref, _true_ref_len);	

	//blend in 1 second 
	_transition_start_time = ros::Time::now().toSec();
	_transition_duration = 1.0;
	_in_transition = true;

	//get closest pt i.e next pt and start blending from there	

	_blend_new_s = std::ceil(_blend_traj_curr_s/resolution);	

//	_mpc_core->set_trajectory(_requested_ss, _requested_xs, _requested_ys);


	//new goal position 
	//_x_goal_euclid = _requested_xs.back();
	//_y_goal_euclid = _requested_ys.back();

	_ref = _requested_ref;
	_ref_len = _requested_len;

	_mpc_core -> set_trajectory(_requested_ss, _requested_xs, _requested_ys);
	}

	publishReference(_requested_ref, _true_len);

	return true;
}
*/


void removePts(Eigen::RowVectorXd*& xs, std::vector<unsigned int> duplicate_pts){



	std::unordered_set<unsigned int> duplicates_set(duplicate_pts.begin(), duplicate_pts.end());
	
	unsigned int new_size = xs->size() - duplicates_set.size();

	unsigned int j = 0;

	
	Eigen::RowVectorXd* dummy = new Eigen::RowVectorXd(new_size);
	for(unsigned int i=0; i<xs->size(); i++){


		if(duplicates_set.count(i)){
			continue;
		}

		(*dummy)(j++) = (*xs)(i);

	}


	delete xs;

	xs = dummy;

}


void reparam_curve(Eigen::RowVectorXd*& xs, Eigen::RowVectorXd*& ys, Eigen::RowVectorXd*& ss){




}


bool MPCCROS::modifyTrajSrv(uvatraj_msgs::ExecuteTraj::Request &req, uvatraj_msgs::ExecuteTraj::Response &res)
{

	if (_is_executing){

		printf("Executing trajectory blend.");	
		_old_ref = _ref;
		_old_ref_len = _ref_len;
		_blend_traj_curr_s = get_s_from_state(_ref, _true_ref_len);


		 printf("Current s from state: %.3f", _blend_traj_curr_s);
		_transition_start_time = ros::Time::now().toSec();
		_transition_duration = 1.0;
	 	_in_transition = true;

		_blend_new_s = std::ceil(_blend_traj_curr_s/resolution);

		printf("Blend target s: %.3f", _blend_new_s);	
	}

	_prev_ref     = _ref;
    	_prev_ref_len = _true_ref_len;	

	int N = req.ctrl_pts.size();

	Eigen::RowVectorXd* ss = new Eigen::RowVectorXd(N);
    	Eigen::RowVectorXd* xs = new Eigen::RowVectorXd(N);
        Eigen::RowVectorXd* ys = new Eigen::RowVectorXd(N);	

	(*xs)(0) = req.ctrl_pts[0].x;
    	(*ys)(0) = req.ctrl_pts[0].y;
	(*ss)(0) = 0.0;
     	for (int i = 1; i < N; ++i) {
    		(*xs)(i) = req.ctrl_pts[i].x;
    		(*ys)(i) = req.ctrl_pts[i].y;
		
		//_prediction_x(i) = req.ctrl_pts[i].x;
		//_prediction_y(i) = req.ctrl_pts[i].y;

		//_traj_executed.push_back(Eigen::Vector2d(req.ctrl_pts[i].x,req.ctrl_pts.y))


    		double dx = (*xs)(i) - (*xs)(i-1);
    		double dy = (*ys)(i) - (*ys)(i-1);
    		(*ss)(i) =(*ss)(i-1) + std::hypot(dx, dy);
	}


    	_ref_len = (*ss)(N - 1);
    	_true_ref_len = _ref_len;
	


	std::vector<unsigned int> duplicate_pts;

	bool valid = true;
	for (int i = 1; i < ss->size(); ++i) {
    		if ((*ss)(i) <= (*ss)(i-1)) {
       			ROS_WARN_STREAM("Non-increasing arc length at i=" << i << ": ss(i)=" << (*ss)(i) << ", ss(i-1)=" << (*ss)(i-1));
    			duplicate_pts.push_back(i);
		} 
	}
	

	
	removePts(xs, duplicate_pts);
	removePts(ys, duplicate_pts);
	removePts(ss, duplicate_pts);
	
	std::vector<double> reparam_x, reparam_y, reparam_z; 

	_traj_executed.resize(xs->size());
	for (int i = 0; i<xs->size(); ++i){	
		_traj_executed[i] = Eigen::Vector2d((*xs)(i),(*ys)(i));
	}

	reparam_curve(xs,ys,ss);

    	const auto fitX = utils::Interp(*xs, 3, *ss);
    	Spline1D splineX(fitX);

    	const auto fitY = utils::Interp(*ys, 3, *ss);
    	Spline1D splineY(fitY);
    	
	_ref[0] = splineX;
    	_ref[1] = splineY;

	

    	ROS_INFO_STREAM("--- Calculated xs, ys, ss before spline fitting ---");
    	for (int i=0; i<N; ++i) {
        	ROS_INFO_STREAM("Point " << i << ": s=" << (*ss)(i) << ", x=" << (*xs)(i) << ", y=" << (*ys)(i));
    	}



    	ROS_INFO_STREAM("--- Sampling SplineX (_ref[0]) and SplineY (_ref[1]) ---");
    	ROS_INFO_STREAM("Spline Domain (arc length s) goes from 0 to " << _ref_len);	

    int num_samples_to_print = 20; // Or more/less as needed
    if (_ref_len <= 0.0 || N == 0) { // N from ctrl_pts.size()
        ROS_WARN_STREAM("Spline length is zero or no control points, cannot sample meaningfully.");
    } else {
        double s_step;
        if (num_samples_to_print > 1) {
            s_step = _ref_len / (num_samples_to_print -1) ;
        } else { // Handle case of wanting to print just 1 sample (e.g., at s=0)
            s_step = _ref_len; // Effectively will loop once for s=0
            if (num_samples_to_print == 0) num_samples_to_print =1; //ensure at least one sample at s=0
        }


        for (int i = 0; i < num_samples_to_print; ++i) {
            double current_s;
            if (num_samples_to_print > 1) {
                 current_s = i * s_step;
            } else {
                current_s = 0.0; // For a single sample, print at s=0
            }
            
            // Ensure current_s does not exceed _ref_len due to floating point issues
            if (current_s > _ref_len) {
                current_s = _ref_len;
            }

            // Evaluate splines. Assuming Spline1D has an operator() or a method like .eval()
            // and that it returns something from which .coeff(0) can get the value.
            // Adjust if your Spline1D API is different.
            double x_val = _ref[0](current_s).coeff(0); // Or splineX(current_s).coeff(0)
            double y_val = _ref[1](current_s).coeff(0); // Or splineY(current_s).coeff(0)

            // Optionally, print derivatives
            // double dx_ds_val = _ref[0].derivatives(current_s, 1).coeff(1);
            // double dy_ds_val = _ref[1].derivatives(current_s, 1).coeff(1);

            ROS_INFO_STREAM("Sample " << i << ": s=" << std::fixed << std::setprecision(3) << current_s
                                     << ", x(s)=" << x_val
                                     << ", y(s)=" << y_val);
                                     // << ", dx/ds=" << dx_ds_val
                                     // << ", dy/ds=" << dy_ds_val);
            if (num_samples_to_print == 1) break; // if only one sample requested
        }
        // Special print for the very end if not perfectly covered by step
        if (num_samples_to_print > 1 && ( (num_samples_to_print-1) * s_step < _ref_len - 1e-5) ) {
             double x_val_end = _ref[0](_ref_len).coeff(0);
             double y_val_end = _ref[1](_ref_len).coeff(0);
             ROS_INFO_STREAM("Sample END: s=" << std::fixed << std::setprecision(3) << _ref_len
                                     << ", x(s)=" << x_val_end
                                     << ", y(s)=" << y_val_end);
        }
    }
    ROS_INFO_STREAM("------------------------------------------------------");

	
    	_mpc_core->set_trajectory(_ref, _ref_len);
	publishReference();
    	visualizeTraj();

	return true;
}

// store in _ref
//double checked logic is sound
bool MPCCROS::generateTrajSrv(uvatraj_msgs::RequestTraj::Request &req, uvatraj_msgs::RequestTraj::Response &res){
	
	//lambda for repetitive code
	auto response = [&](const std::string_view& msg, const bool& success) -> void{
		res.success = success;
		res.status_message = msg;
		if (!success)
			ROS_ERROR_STREAM(msg);
	};
	

	//Errors
	if (!_is_init){
		response("NO POSITIONAL DATA", false);
		return true;
	}
	
	std::vector<Eigen::Vector2d> ctrl_pts;
	
 	ROS_DEBUG_STREAM("[vicon_bridge] Initial ctrl_pts:"
                 << " start=(" << _odom(0) << "," << _odom(1) << ")"
                 << " goal=("  << -req.goal.z << "," << req.goal.y << ")");
	
	ctrl_pts.emplace_back(Eigen::Vector2d(_odom(0),_odom(1)));
	ctrl_pts.emplace_back(Eigen::Vector2d(-req.goal.z,req.goal.y));
	ctrl_pts = mpcc::arutils::generateLinearTrajectory(ctrl_pts[0], ctrl_pts[1], resolution);
	_traj_executed = ctrl_pts;
	uvatraj_msgs::ControlPoint holder;
	
	ROS_DEBUG_STREAM("[vicon_bridge] Generated trajectory: "
                 << ctrl_pts.size() << " points"
                 << " @ resolution=" << resolution);


	for (int i=0; i<ctrl_pts.size(); ++i){
	
		holder.x = ctrl_pts[i].x();
		holder.y = ctrl_pts[i].y();

		//_prediction_x(i)=ctrl_pts[i].x();
		//_prediction_y(i)=ctrl_pts[i].y();
		
		holder.z = 0.0;
		holder.metadata = "";

		if (i%5==0) res.boundary_ctrl_pts.push_back(holder);
		res.all_ctrl_pts.push_back(holder);
			
		

	}

	response("SUCCESSFULLY GENERATED PATH", true);
		
	//set trajectory	
	
	 _prev_ref     = _ref;
    	_prev_ref_len = _true_ref_len;

    int N = ctrl_pts.size();

    Eigen::RowVectorXd ss, xs, ys;
    ss.resize(N);
    xs.resize(N);
    ys.resize(N);
/*
    for (int i = 0; i < N; ++i)
    {
        xs(i) = ctrl_pts[i].x();
        ys(i) = ctrl_pts[i].y();
        ss(i) = 0;

    }
*/


    xs(0) = ctrl_pts[0].x();
    ys(0) = ctrl_pts[0].y();
     for (int i = 1; i < N; ++i) {
    	xs(i) = ctrl_pts[i].x();
    	ys(i) = ctrl_pts[i].y();

    	double dx = xs(i) - xs(i-1);
    	double dy = ys(i) - ys(i-1);
    	ss(i) = ss(i-1) + std::hypot(dx, dy);
	}
    _ref_len = ss(N - 1);

    _ref_len      = ss(ss.size() - 1);
    _true_ref_len = _ref_len;

    const auto fitX = utils::Interp(xs, 3, ss);
    Spline1D splineX(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    Spline1D splineY(fitY);

    // if reference length is less than required mpc size, extend trajectory
    if (_ref_len < _mpc_ref_len_sz){
     ROS_WARN("reference length (%.2f) is smaller than %.2fm, extending", _ref_len,
                 _mpc_ref_len_sz);	
	}

    _ref[0] = splineX;
    _ref[1] = splineY;


    ROS_INFO_STREAM("--- Calculated xs, ys, ss before spline fitting ---");
    for (int i=0; i<N; ++i) {
        ROS_INFO_STREAM("Point " << i << ": s=" << ss(i) << ", x=" << xs(i) << ", y=" << ys(i));
    }
    ROS_INFO_STREAM("Total calculated _ref_len (before extension): " << _ref_len);
    ROS_INFO_STREAM("---------------------------------------------------"); 


        ROS_INFO("**********************************************************");
    ROS_INFO("MPC generateTrajSrv: Trajectory set! Original Length (_true_ref_len): %.2f, Effective MPC Length (_ref_len): %.2f", _true_ref_len, _ref_len);
    ROS_INFO("**********************************************************");

    // --- Add Spline Printing/Sampling Logic Here ---
    ROS_INFO_STREAM("--- Sampling SplineX (_ref[0]) and SplineY (_ref[1]) ---");
    ROS_INFO_STREAM("Spline Domain (arc length s) goes from 0 to " << _ref_len);

    int num_samples_to_print = 20; // Or more/less as needed
    if (_ref_len <= 0.0 || N == 0) { // N from ctrl_pts.size()
        ROS_WARN_STREAM("Spline length is zero or no control points, cannot sample meaningfully.");
    } else {
        double s_step;
        if (num_samples_to_print > 1) {
            s_step = _ref_len / (num_samples_to_print -1) ;
        } else { // Handle case of wanting to print just 1 sample (e.g., at s=0)
            s_step = _ref_len; // Effectively will loop once for s=0
            if (num_samples_to_print == 0) num_samples_to_print =1; //ensure at least one sample at s=0
        }


        for (int i = 0; i < num_samples_to_print; ++i) {
            double current_s;
            if (num_samples_to_print > 1) {
                 current_s = i * s_step;
            } else {
                current_s = 0.0; // For a single sample, print at s=0
            }
            
            // Ensure current_s does not exceed _ref_len due to floating point issues
            if (current_s > _ref_len) {
                current_s = _ref_len;
            }

            // Evaluate splines. Assuming Spline1D has an operator() or a method like .eval()
            // and that it returns something from which .coeff(0) can get the value.
            // Adjust if your Spline1D API is different.
            double x_val = _ref[0](current_s).coeff(0); // Or splineX(current_s).coeff(0)
            double y_val = _ref[1](current_s).coeff(0); // Or splineY(current_s).coeff(0)

            // Optionally, print derivatives
            // double dx_ds_val = _ref[0].derivatives(current_s, 1).coeff(1);
            // double dy_ds_val = _ref[1].derivatives(current_s, 1).coeff(1);

            ROS_INFO_STREAM("Sample " << i << ": s=" << std::fixed << std::setprecision(3) << current_s
                                     << ", x(s)=" << x_val
                                     << ", y(s)=" << y_val);
                                     // << ", dx/ds=" << dx_ds_val
                                     // << ", dy/ds=" << dy_ds_val);
            if (num_samples_to_print == 1) break; // if only one sample requested
        }
        // Special print for the very end if not perfectly covered by step
        if (num_samples_to_print > 1 && ( (num_samples_to_print-1) * s_step < _ref_len - 1e-5) ) {
             double x_val_end = _ref[0](_ref_len).coeff(0);
             double y_val_end = _ref[1](_ref_len).coeff(0);
             ROS_INFO_STREAM("Sample END: s=" << std::fixed << std::setprecision(3) << _ref_len
                                     << ", x(s)=" << x_val_end
                                     << ", y(s)=" << y_val_end);
        }
    }
    ROS_INFO_STREAM("------------------------------------------------------");


    _mpc_core->set_trajectory(_ref, _ref_len);
publishReference();
    visualizeTraj();

    ROS_INFO("**********************************************************");
    ROS_INFO("MPC received trajectory! Length: %.2f", _ref_len);
    ROS_INFO("**********************************************************");
	
	return true;

	
}


//definitely correct
void MPCCROS::viconcb(const geometry_msgs::TransformStamped::ConstPtr& data){


	/*ROS_ERROR("Received VICON transform: [x=%.3f, y=%.3f, z=%.3f] [qw=%.3f, qx=%.3f, qy=%.3f, qz=%.3f]",
              data->transform.translation.x,
              data->transform.translation.y,
              data->transform.translation.z,
              data->transform.rotation.w,
              data->transform.rotation.x,
              data->transform.rotation.y,
              data->transform.rotation.z);
*/


//	tf::Quaternion q(data->transform.translation.x, data->transform.translation.y, data->transform.translation.z, data->transform.rotation.w);

	//tf::Quaternion q_tf;

	//tf::quaternionMsgToTF(data->transform.rotation, q_tf);

	
	
	//tf::Matrix3x3 m(q_tf);

	//double roll, pitch, yaw;

	//m.getRPY(roll,pitch,yaw);

	tf::Quaternion q(
        data->transform.rotation.x,
        data->transform.rotation.y,
        data->transform.rotation.z,
        data->transform.rotation.w);

    // Convert quaternion to roll, pitch, yaw
    	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(roll, pitch, yaw);

	//ROS_ERROR("Converted RPY: roll=%.3f, pitch=%.3f, yaw=%.3f", roll, pitch, yaw);


	_odom = Eigen::VectorXd(3);

	_odom(0) = data->transform.translation.x;
	_odom(1) = data->transform.translation.y;
	_odom(2) = yaw;

	
    	if (_reverse_mode)
    	{
        	_odom(2) += M_PI;
        // wrap to pi
        	if (_odom(2) > M_PI) _odom(2) -= 2 * M_PI;
		else if (_odom(2) < -M_PI) _odom(2) += 2*M_PI;
    	}

    	_mpc_core->set_odom(_odom);

    	if (!_is_init)
    	{
        	_is_init = true;
        	ROS_INFO("tracker initialized");
    	}


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
        pt_a.z                     = 1.0;

        std_msgs::ColorRGBA color_msg;
        color_msg.r = 0;
        color_msg.g = 0.0;
        color_msg.b = 192. / 255.;
        color_msg.a = 1.0;

        traj.colors.push_back(color_msg);
    }

    _refVizPub.publish(traj);
}

void MPCCROS::publishVel()
{
    constexpr double pub_vel_loop_rate_hz = 50;
    //const std::chrono::milliseconds pub_loop_period(
      ///  static_cast<int>(1000.0 / pub_vel_loop_rate_hz));


	ros::Rate loop_rate(pub_vel_loop_rate_hz);
    while (ros::ok())
    {
        //if (_trajectory.points.size() > 0) _velPub.publish(_vel_msg);
	if(_is_init){
        	 _velPub.publish(_vel_msg);
	}
        //std::this_thread::sleep_for(pub_loop_period);
        loop_rate.sleep();
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
void MPCCROS::trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr& msg)
{
    ROS_INFO("Trajectory received!");
    _trajectory = *msg;

    if (msg->points.size() == 0)
    {
        ROS_WARN("Trajectory is empty, stopping!");
        _vel_msg.linear.x  = 0;
        _vel_msg.angular.z = 0;
        return;
    }

    _prev_ref     = _ref;
    _prev_ref_len = _true_ref_len;

    _traj_reset = true;

    int N = msg->points.size();

    Eigen::RowVectorXd ss, xs, ys;
    ss.resize(N);
    xs.resize(N);
    ys.resize(N);

    for (int i = 0; i < N; ++i)
    {
        xs(i) = msg->points[i].positions[0];
        ys(i) = msg->points[i].positions[1];
        ss(i) = msg->points[i].time_from_start.toSec();

        /*ROS_INFO("%.2f:\t(%.2f, %.2f)", ss(i), xs(i), ys(i));*/
    }

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

    _mpc_core->set_trajectory(_ref, _ref_len);

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
        double old_x = _old_ref[0](s_old).coeff(0);
        double old_y = _old_ref[1](s_old).coeff(0);
        double new_x = _ref[0](s_new).coeff(0);
        double new_y = _ref[1](s_new).coeff(0);
        
        // Linear interpolation between trajectories
        double blended_x = old_x * (1 - blend_factor) + new_x * blend_factor;
        double blended_y = old_y * (1 - blend_factor) + new_y * blend_factor;
        
        // Scale s coordinate for the blended trajectory
        double s_blended = t * std::min(_old_ref_len, _ref_len);
        _mpc_core->updateReferencePoint(s_blended, blended_x, blended_y);
    }
}


void MPCCROS::sendTraj(){

   uvatraj_msgs::ExecuteTraj msg;
   uvatraj_msgs::ControlPoint pt;

   for(int i = 0; i<_traj_executed.size(); ++i){

	   pt.x = _traj_executed[i][0];
	   pt.y = _traj_executed[i][1];
	   msg.request.ctrl_pts.push_back(pt);

   } 
   if (_traj_sender.call(msg)){
	ROS_INFO("sendTraj: success=%s, msg=%s",
                 msg.response.success ? "true" : "false",
                 msg.response.status_message.c_str());
		   }else{
	ROS_ERROR("Failed to call traj_send service");
	}
}


void MPCCROS::mpcc_ctrl_loop(const ros::TimerEvent& event)
{
    if (!_is_init || _estop){


	    _vel_msg.linear.x = 0;
	    _vel_msg.angular.z = 0;
	    return;
    }

    //if (_trajectory.points.size() == 0) return;

    if(!_is_executing){


	    _vel_msg.linear.x = 0;
	    _vel_msg.angular.z = 0;
	    return;
    }

    if (_ref_len <= 1e-3){


	    ROS_WARN("MPC Control Loop: invalid traj length %.2f", _ref_len); 
	    _vel_msg.linear.x = 0;
	    _vel_msg.angular.z = 0;
	    _is_executing = false;
	    return;
    }


    if (_in_transition){

	double current_time = ros::Time::now().toSec();
	double elapsed_time = _transition_start_time - current_time;
	//double blend_factor = (current_time - _transition_start_time)/_transition_duration;
	double blend_factor = 0.5 * (1 - cos(M_PI * elapsed_time / _transition_duration));
	if (blend_factor >= 1.0){
	    _in_transition = false;
	    ROS_INFO("Trajectory blend finished");
    	} else {
	    ROS_INFO("Blending trajectories: %.1f%%", blend_factor*100);
	    blendTrajectories(blend_factor);
	}
    }

    if(_is_paused){
	    ROS_INFO_STREAM("paused");
	    _vel_msg.linear.x = 0.0;
	    _vel_msg.angular.z = 0.0;
	    return;
    }

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

    ROS_INFO("S DOT IS: %.2f", _s_dot);
    ROS_INFO("corrected len is: %.2f / %.2f", corrected_len, _true_ref_len);
    ROS_INFO("prev_s is: %.2f", _prev_s);
    ROS_INFO("corrected len - prev_s / dt is %.2f", (corrected_len - _prev_s) / _dt);

    if (len_start > _true_ref_len - 3e-1)
    {
        ROS_INFO("Reached end of traj %.2f / %.2f", len_start, _true_ref_len);
        _vel_msg.linear.x  = 0;
        _vel_msg.angular.z = 0;
	_is_executing = false;
	_traj_reset = false;
        _trajectory.points.clear();
	sendTraj();
        return;
    }

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
        visualizeTubes();

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
    double vel = _vel_msg.linear.x;
    if (_reverse_mode) vel *= -1;
    state << _odom(0), _odom(1), _odom(2), vel, 0, _s_dot;

    std::array<double, 2> mpc_results = _mpc_core->solve(state, _reverse_mode);

    _vel_msg.linear.x  = mpc_results[0];
    _vel_msg.angular.z = mpc_results[1];

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
        tmp.pose.position.z    = .1;
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
