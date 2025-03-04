#pragma once

#include <string>
#include <thread>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseArray.h>
#include <uvatraj_msgs/ExecuteTraj.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <costmap_2d/costmap_2d_ros.h>

#include "uav_mpc/mpcc_core.h"


class MPCCROS
{
public:
	MPCCROS(ros::NodeHandle &nh);
	~MPCCROS();

	void LoadParams(const std::map<std::string, double> &params);

private:
	ros::Subscriber _distMapSub;
	ros::Subscriber _trajSub;
	ros::Subscriber _trajNoResetSub;
	ros::Subscriber _obsSub;
	ros::Subscriber _alphaSub;
	ros::Subscriber _odomSub;
	ros::Subscriber _viconSub;
	ros::Subscriber _splinePathSub;
	ros::Subscriber _collisionSub;
	ros::Subscriber _mapSub;

	ros::Publisher _velPub;
	ros::Publisher _trajPub;
	ros::Publisher _pathPub;
	ros::Publisher _pointPub;
	ros::Publisher _odomPub;
	ros::Publisher _refPub;
	ros::Publisher _goalReachedPub;
	ros::Publisher _horizonPub;
	ros::Publisher _solveTimePub;
	ros::Publisher _donePub;
	ros::Publisher _loggingPub;
	ros::Publisher _tubeVizPub;

	ros::ServiceServer _eStop_srv;
	ros::ServiceServer _mode_srv;
	ros::ServiceServer _modify_traj_srv;
	ros::ServiceServer _execute_traj_srv;

	ros::ServiceClient _sac_srv;

	ros::NodeHandle _nh;

	ros::Timer _timer, _velPubTimer;

	Eigen::VectorXd _odom;

	trajectory_msgs::JointTrajectory trajectory;
	trajectory_msgs::JointTrajectoryPoint current_reference;

	costmap_2d::Costmap2DROS *_local_costmap;

	std::vector<Eigen::Vector3d> poses;
	std::vector<double> mpc_results;
	std::vector<SplineWrapper> _ref;
	std::vector<SplineWrapper> _requested_ref;
	std::vector<double> _requested_ss;
	std::vector<double> _requested_xs;
	std::vector<double> _requested_ys;

	// MPCBase* _mpc;
	std::unique_ptr<MPCCore> _mpc_core;
	std::map<std::string, double> _mpc_params;

	double _x_goal_euclid = 0;
	double _y_goal_euclid =0;

	double _mpc_steps, _w_vel, _w_angvel, _w_linvel, _w_angvel_d, _w_linvel_d, _w_etheta,
		_max_angvel, _max_linvel, _bound_value, _x_goal, _y_goal, _theta_goal, _tol,
		_max_linacc, _max_anga, _w_cte, _w_pos;

	double _cbf_alpha, _cbf_colinear, _cbf_padding;

	double _prop_gain, _prop_angle_thresh;

	double _min_alpha;
	double _max_alpha;
	double _ref_len;
	double _true_len;
	double _requested_len;

	const int XI = 0;
	const int YI = 1;
	const int THETAI = 2;

	double _dt, _curr_vel, _curr_ang_vel, _vel_pub_freq;
	bool _is_init, _is_goal, _teleop, _traj_reset, _use_vicon, _estop,
		_is_at_goal, _use_cbf, _use_dynamic_alpha;
	bool _is_executing;

	Eigen::MatrixX4d _poly;
	geometry_msgs::Twist velMsg;

	Eigen::VectorXd _prev_rl_state;
	Eigen::VectorXd _curr_rl_state;

	std::string _frame_id;
	std::string _logging_table_name;
	std::string _logging_topic_name;

	std::thread timer_thread;

	std::shared_ptr<distmap::DistanceMap> _dist_grid_ptr;

	void publishMPCTrajectory();
	void publishReference(const std::vector<SplineWrapper>& ref, double ref_len);


	bool modifyTrajSrv(uvatraj_msgs::ExecuteTraj::Request &req, 
					   uvatraj_msgs::ExecuteTraj::Response &res);
	
	bool executeTrajSrv(std_srvs::Empty::Request &req, 
						std_srvs::Empty::Response &res);


	void cte_ctrl_loop();
	void pos_ctrl_loop();

	double get_s_from_state(const std::vector<SplineWrapper>& ref, double ref_len);

	void alphacb(const std_msgs::Float64::ConstPtr &msg);
	void odomcb(const nav_msgs::Odometry::ConstPtr &msg);
	void viconCb(const geometry_msgs::TransformStamped::ConstPtr& msg);
	void goalcb(const geometry_msgs::PoseStamped::ConstPtr &msg);
	void distmapcb(const distance_map_msgs::DistanceMap::ConstPtr& msg);
	void trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr &msg);

	// void publishVel(const ros::TimerEvent&);
	void publishVel();
	void controlLoop(const ros::TimerEvent &);

	//for smooth transitions between trajectories
	double _transition_start_time;
    double _transition_duration;
    bool _in_transition;
    std::vector<SplineWrapper> _old_ref;
    double _old_ref_len;
	void splinePathCb(const nav_msgs::Path::ConstPtr &msg);
    void blendTrajectories(double blend_factor);
};
