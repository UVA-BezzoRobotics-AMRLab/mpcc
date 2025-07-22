#include <mpcc/logger.h>
#include <std_msgs/Float64.h>

#include <Eigen/Core>
#include <string>

namespace logger {

RLLogger::RLLogger(ros::NodeHandle& nh, double min_alpha, double max_alpha,
                   bool is_logging) {
  _nh         = nh;
  _min_alpha  = min_alpha;
  _max_alpha  = max_alpha;
  _is_logging = is_logging;

  _table_name = "replay_buffer";
  _topic_name = "/cbf_rl_learning";

  _done_pub      = _nh.advertise<std_msgs::Bool>("/mpc_done", 100);
  _alpha_pub_abv = _nh.advertise<std_msgs::Float64>("/cbf_alpha_abv", 100);
  _alpha_pub_blw = _nh.advertise<std_msgs::Float64>("/cbf_alpha_blw", 100);
  _logging_pub   = _nh.advertise<amrl_logging::LoggingData>(_topic_name, 100);

  _collision_sub =
      _nh.subscribe("/collision", 1, &RLLogger::collision_cb, this);

  _sac_srv = nh.serviceClient<mpcc::QuerySAC>("/query_sac");

  _count         = 0;
  _is_done       = false;
  _is_first_iter = true;
  _is_colliding  = false;

  _exceeded_bounds = 0;

  _alpha_dot_abv = 0.;
  _alpha_dot_blw = 0.;

  const std::vector<std::string> string_types(
      {"prev_solver_status", "curr_solver_status", "is_done"});
  const std::vector<std::string> float_types({"id",
                                              "prev_theta",
                                              "prev_vel",
                                              "prev_acc",
                                              "prev_angvel",
                                              "prev_obs_dist_abv",
                                              "prev_obs_dist_blw",
                                              "prev_obs_heading",
                                              "prev_progress",
                                              "prev_h_abv",
                                              "prev_h_blw",
                                              "prev_alpha_abv",
                                              "prev_alpha_blw",
                                              "alpha_dot_abv",
                                              "alpha_dot_blw",
                                              "reward",
                                              "curr_theta",
                                              "curr_vel",
                                              "curr_acc",
                                              "curr_angvel",
                                              "curr_obs_dist_abv",
                                              "curr_obs_dist_blw",
                                              "curr_obs_heading",
                                              "curr_progress",
                                              "curr_h_abv",
                                              "curr_h_blw",
                                              "curr_alpha_abv",
                                              "curr_alpha_blw"});

  if (_is_logging && !amrl::logging_setup(_nh, _table_name, _topic_name,
                                          string_types, {}, float_types)) {
    ROS_ERROR("[Logger] Failed to setup logging");
    exit(-1);
  }
}

RLLogger::~RLLogger() {
  amrl::logging_finish(_nh, _table_name);
}

void RLLogger::collision_cb(const std_msgs::Bool::ConstPtr& msg) {
  _is_colliding = msg->data;
}

bool RLLogger::request_alpha(MPCCore& mpc_core, double ref_len) {
  Eigen::VectorXd mpc_state       = mpc_core.get_state();
  std::array<double, 2> mpc_input = mpc_core.get_mpc_command();
  bool solver_status              = mpc_core.get_solver_status();

  Eigen::VectorXd cbf_data_abv = mpc_core.get_cbf_data(
      mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
  Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
      mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

  // double curr_progress = 1;
  // if (ref_len > 1e-3) curr_progress = mpc_state[4] / ref_len;
  // if (curr_progress > 1.) curr_progress = 1.;
  double max_vel       = mpc_core.get_params().at("LINVEL");
  double curr_progress = mpc_state[5] / max_vel;

  mpcc::QuerySAC req;
  req.request.theta         = mpc_state[2];
  req.request.vel           = mpc_state[3];
  req.request.acc           = mpc_input[1];
  req.request.ang_vel       = mpc_input[0];
  req.request.obs_dist_abv  = cbf_data_abv[1];
  req.request.obs_dist_blw  = cbf_data_blw[1];
  req.request.heading_dist  = cbf_data_abv[2];
  req.request.progress      = curr_progress;
  req.request.h_val_abv     = cbf_data_abv[0];
  req.request.h_val_blw     = cbf_data_blw[0];
  req.request.alpha_abv     = mpc_core.get_params().at("CBF_ALPHA_ABV");
  req.request.alpha_blw     = mpc_core.get_params().at("CBF_ALPHA_BLW");
  req.request.solver_status = solver_status;

  if (_sac_srv.call(req)) {
    if (!req.response.success)
      ROS_ERROR("SAC service failed");
    // integrate alpha_dot into CBF_ALPHA
    // clip alpha to ensure it's within bounds
    std::map<std::string, double> mpc_params = mpc_core.get_params();
    double dt                                = mpc_params.at("DT");

    _alpha_dot_abv = req.response.alpha_dot[0];
    _alpha_dot_blw = req.response.alpha_dot[1];

    double alpha_abv = mpc_params["CBF_ALPHA_ABV"] + _alpha_dot_abv * dt;
    double alpha_blw = mpc_params["CBF_ALPHA_BLW"] + _alpha_dot_blw * dt;

    if (alpha_abv < _min_alpha || alpha_abv > _max_alpha)
      _exceeded_bounds++;
    if (alpha_blw < _min_alpha || alpha_blw > _max_alpha)
      _exceeded_bounds++;

    alpha_abv = std::max(_min_alpha, std::min(_max_alpha, alpha_abv));
    alpha_blw = std::max(_min_alpha, std::min(_max_alpha, alpha_blw));

    std_msgs::Float64 alpha_msg;
    alpha_msg.data = alpha_abv;
    _alpha_pub_abv.publish(alpha_msg);

    std_msgs::Float64 alpha_msg_blw;
    alpha_msg_blw.data = alpha_blw;
    _alpha_pub_blw.publish(alpha_msg_blw);

    mpc_params["CBF_ALPHA_ABV"] = alpha_abv;
    mpc_params["CBF_ALPHA_BLW"] = alpha_blw;
    mpc_core.load_params(mpc_params);
  } else {
    ROS_ERROR("Failed to call service query_sac");
    return false;
  }

  return true;
}

void RLLogger::log_transition(const MPCCore& mpc_core, double len_start,
                              double ref_len) {
  if (_is_first_iter) {
    Eigen::VectorXd mpc_state       = mpc_core.get_state();
    std::array<double, 2> mpc_input = mpc_core.get_mpc_command();
    bool solver_status              = mpc_core.get_solver_status();

    Eigen::VectorXd cbf_data_abv = mpc_core.get_cbf_data(
        mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
    Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
        mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

    double alpha_abv = mpc_core.get_params().at("CBF_ALPHA_ABV");
    double alpha_blw = mpc_core.get_params().at("CBF_ALPHA_BLW");

    // double curr_progress = 1;
    // if (ref_len > 1e-3) curr_progress = mpc_state[4] / ref_len;
    // if (curr_progress > 1.) curr_progress = 1.;

    double max_vel       = mpc_core.get_params().at("LINVEL");
    double curr_progress = mpc_state[5] / max_vel;

    _curr_rl_state.theta         = mpc_state[2];
    _curr_rl_state.vel           = mpc_state[3];
    _curr_rl_state.obs_dist_abv  = cbf_data_abv[1];
    _curr_rl_state.obs_dist_blw  = cbf_data_blw[1];
    _curr_rl_state.obs_heading   = cbf_data_abv[2];
    _curr_rl_state.progress      = curr_progress;
    _curr_rl_state.h_val_abv     = cbf_data_abv[0];
    _curr_rl_state.h_val_blw     = cbf_data_blw[0];
    _curr_rl_state.alpha_val_abv = alpha_abv;
    _curr_rl_state.alpha_val_blw = alpha_blw;
    _curr_rl_state.ang_vel       = mpc_input[0];
    _curr_rl_state.acc           = mpc_input[1];
    _curr_rl_state.solver_status = solver_status;

    _prev_rl_state = _curr_rl_state;

    _is_first_iter = false;
  }
  // we don't want to log if already reported an is_done state
  else if (!_is_done) {
    double reward = 0;

    const Eigen::VectorXd& mpc_state = mpc_core.get_state();
    std::array<double, 2> mpc_input  = mpc_core.get_mpc_command();
    bool solver_status               = mpc_core.get_solver_status();

    Eigen::VectorXd cbf_data_abv = mpc_core.get_cbf_data(
        mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
    Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
        mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

    double alpha_abv = mpc_core.get_params().at("CBF_ALPHA_ABV");
    double alpha_blw = mpc_core.get_params().at("CBF_ALPHA_BLW");

    // double curr_progress = 1;
    // if (ref_len > 1e-3) curr_progress = mpc_state[4] / ref_len;
    // if (curr_progress > 1.) curr_progress = 1.;
    double max_vel       = mpc_core.get_params().at("LINVEL");
    double curr_progress = mpc_state[5] / max_vel;

    _curr_rl_state.theta         = mpc_state[2];
    _curr_rl_state.vel           = mpc_state[3];
    _curr_rl_state.obs_dist_abv  = cbf_data_abv[1];
    _curr_rl_state.obs_dist_blw  = cbf_data_blw[1];
    _curr_rl_state.obs_heading   = cbf_data_abv[2];
    _curr_rl_state.progress      = curr_progress;
    _curr_rl_state.h_val_abv     = cbf_data_abv[0];
    _curr_rl_state.h_val_blw     = cbf_data_blw[0];
    _curr_rl_state.alpha_val_abv = alpha_abv;
    _curr_rl_state.alpha_val_blw = alpha_blw;
    _curr_rl_state.ang_vel       = mpc_input[0];
    _curr_rl_state.acc           = mpc_input[1];
    _curr_rl_state.solver_status = solver_status;

    if (!_is_colliding) {
      // weight distance to obstacle
      reward = 5 * _curr_rl_state.obs_dist_abv * _curr_rl_state.obs_dist_blw;
    } else {
      _is_done = true;
    }

    // add penalty for not making progress
    // reward -= 12 * (1 - _curr_rl_state(6));
    // ref len should never be negative, but fabs just in case
    // need to use passed in len-start because core version is relative

    // potentially use velocity along the path as the reward here!
    // reward -= 12 * (1 - curr_progress);
    reward -= 5 * (1 - curr_progress);

    // add small penalty for large alpha jumps
    // reward -= 0.1 * _alpha_dot_abv * _alpha_dot_abv;
    // reward -= 0.1 * _alpha_dot_blw * _alpha_dot_blw;

    // add penalty for using higher alpha values
    // reward -= .1 * (_curr_rl_state(8)- _min_alpha);

    // if alpha value is outside bounds, penalize heavily
    // penalize linearly as alpha_abv approaches max/min alpha
    double mid_alpha = (_max_alpha + _min_alpha) / 2.0;
    reward -= 5 * (_curr_rl_state.alpha_val_abv - mid_alpha) *
              (_curr_rl_state.alpha_val_abv - mid_alpha);

    reward -= 5 * (_curr_rl_state.alpha_val_blw - mid_alpha) *
              (_curr_rl_state.alpha_val_blw - mid_alpha);

    reward -= 30 * _exceeded_bounds;

    // if h_value is negative, penalize heavily
    /*if (_curr_rl_state.h_val_abv < 0) reward -= 20;*/
    /*if (_curr_rl_state.h_val_blw < 0) reward -= 20;*/

    // add reward for h_values being above 0
    if (_curr_rl_state.h_val_abv > 0)
      reward += 7 * _curr_rl_state.h_val_abv;
    if (_curr_rl_state.h_val_blw > 0)
      reward += 7 * _curr_rl_state.h_val_blw;

    if (!_curr_rl_state.solver_status)
      reward -= 25;

    if (_is_done)
      reward -= 25;

    // std::cout << "reward is: " << reward << std::endl;
    // std::cout << "\tprogress: " << -12 * (1 - curr_progress) << std::endl;
    // std::cout << "\talpha_dot_abv: " << -0.1 * _alpha_dot_abv *
    // _alpha_dot_abv << std::endl; std::cout << "\talpha_dot_blw: " << -0.1 *
    // _alpha_dot_blw * _alpha_dot_blw
    // << std::endl; std::cout << "\texceeded bounds: " << -20 *
    // _exceeded_bounds << std::endl; std::cout << "\th_val_abv: " << -10 *
    // (_curr_rl_state.h_val_abv < 0) << std::endl; std::cout << "\th_val_blw: "
    // << -10 * (_curr_rl_state.h_val_blw < 0) << std::endl;

    _exceeded_bounds = 0;

    // log to database
    amrl_logging::LoggingData row;
    std::string is_done_str = _is_done ? "true" : "false";
    std::string prev_solver_stat =
        _prev_rl_state.solver_status ? "true" : "false";
    std::string curr_solver_stat =
        _curr_rl_state.solver_status ? "true" : "false";
    std::vector<std::string> string_data = {prev_solver_stat, curr_solver_stat,
                                            is_done_str};
    std::vector<double> numeric_data     = {
            _count,
            _prev_rl_state.theta,          // theta
            _prev_rl_state.vel,            // velocity
            _prev_rl_state.acc,            // acceleration
            _prev_rl_state.ang_vel,        // angular velocity
            _prev_rl_state.obs_dist_abv,   // distance to obstacle abv
            _prev_rl_state.obs_dist_blw,   // distance to obstacle blw
            _prev_rl_state.obs_heading,    // heading to obstacle
            _prev_rl_state.progress,       // progress
            _prev_rl_state.h_val_abv,      // h value
            _prev_rl_state.h_val_blw,      // h value
            _prev_rl_state.alpha_val_abv,  // alpha value
            _prev_rl_state.alpha_val_blw,  // alpha value
            _alpha_dot_abv,
            _alpha_dot_blw,
            reward,
            _curr_rl_state.theta,           // theta
            _curr_rl_state.vel,             // velocity
            _curr_rl_state.acc,             // acceleration
            _curr_rl_state.ang_vel,         // angular velocity
            _curr_rl_state.obs_dist_abv,    // distance to obstacle abv
            _curr_rl_state.obs_dist_blw,    // distance to osbtacle blw
            _curr_rl_state.obs_heading,     // heading to obstacle
            _curr_rl_state.progress,        // progress
            _curr_rl_state.h_val_abv,       // h value
            _curr_rl_state.h_val_blw,       // h value
            _curr_rl_state.alpha_val_abv,   // alpha value
            _curr_rl_state.alpha_val_blw};  // alpha value

    row.header.seq += _count++;
    row.header.stamp = ros::Time::now();
    row.labels       = string_data;
    row.reals        = numeric_data;

    if (_is_logging)
      _logging_pub.publish(row);

    _prev_rl_state = _curr_rl_state;
  }

  std_msgs::Bool done_msg;
  done_msg.data = _is_done;
  _done_pub.publish(done_msg);
}

}  // namespace logger
