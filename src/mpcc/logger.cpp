#include <mpcc/logger.h>
#include <std_msgs/Float64.h>

#include <Eigen/Core>
#include <iterator>
#include <string>
#include "ros/console.h"

namespace logger {

RLLogger::RLLogger(ros::NodeHandle& nh, double min_alpha, double max_alpha,
                   double max_obs_dist, bool is_logging,
                   const std::string& mpc_type) {
  _nh         = nh;
  _min_alpha  = min_alpha;
  _max_alpha  = max_alpha;
  _is_logging = is_logging;

  _table_name = "replay_buffer";
  _topic_name = "/cbf_rl_learning";

  _mpc_type = mpc_type;

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

  _max_obs_dist = max_obs_dist;

  const std::vector<std::string> string_types = {"prev_state", "action",
                                                 "next_state", "is_done"};

  std::vector<std::string> float_types = {"reward"};

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

  // TODO: Modify the data handling to eventually remove the magic numbers...
  mpcc::QuerySAC req;

  fill_state(mpc_core, req.request.state);

  if (!_sac_srv.call(req)) {
    ROS_ERROR("Failed to call service query_sac");
    return false;
  }

  if (!req.response.success) {
    ROS_ERROR("SAC service failed");
    return false;
  }

  // integrate alpha_dot into CBF_ALPHA
  // clip alpha to ensure it's within bounds
  std::map<std::string, double> mpc_params = mpc_core.get_params();
  double dt                                = mpc_params.at("DT");

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

  return true;
}

void RLLogger::log_transition(const MPCCore& mpc_core, double len_start,
                              double ref_len) {

  fill_state(mpc_core, _curr_rl_state);

  if (_is_first_iter) {
    _prev_rl_state = _curr_rl_state;
    _is_first_iter = false;
  } else if (!_is_done) {
    // we don't want to log if already reported an is_done state
    double reward = compute_reward();

    // log to database
    amrl_logging::LoggingData row;
    std::string is_done_str = _is_done ? "true" : "false";
    std::string prev_solver_stat =
        _prev_rl_state.solver_status ? "true" : "false";
    std::string curr_solver_stat =
        _curr_rl_state.solver_status ? "true" : "false";

    std::vector<std::string> string_data = {
        serialize_state(_prev_rl_state),
        std::to_string(_alpha_dot_abv) + "," + std::to_string(_alpha_dot_blw),
        serialize_state(_curr_rl_state), is_done_str};

    std::vector<double> numeric_data = {reward};

    row.header.seq += _count++;
    row.header.stamp = ros::Time::now();
    row.labels       = string_data;
    row.reals        = numeric_data;

    if (_is_logging)
      _logging_pub.publish(row);

    _prev_rl_state   = _curr_rl_state;
    _exceeded_bounds = 0;
  }

  std_msgs::Bool done_msg;
  done_msg.data = _is_done;
  _done_pub.publish(done_msg);
}

double RLLogger::compute_reward() {
  double reward = 0;
  if (!_is_colliding) {
    // weight distance to obstacle
    reward = 5 * _curr_rl_state.state[4] * _curr_rl_state.state[5];
  } else {
    _is_done = true;
  }

  reward -= 5 * (1 - _curr_rl_state.state[7]);

  // if alpha value is outside bounds, penalize heavily
  // penalize linearly as alpha_abv approaches max/min alpha
  double mid_alpha = (_max_alpha + _min_alpha) / 2.0;
  reward -= 5 * (_curr_rl_state.state[10] - mid_alpha) *
            (_curr_rl_state.state[10] - mid_alpha);

  reward -= 5 * (_curr_rl_state.state[11] - mid_alpha) *
            (_curr_rl_state.state[11] - mid_alpha);

  reward -= 30 * _exceeded_bounds;

  // add reward for h_values being above 0
  if (_curr_rl_state.state[8] > 0)
    reward += 7 * _curr_rl_state.state[8];
  if (_curr_rl_state.state[9] > 0)
    reward += 7 * _curr_rl_state.state[8];

  if (!_curr_rl_state.solver_status)
    reward -= 25;

  if (_is_done)
    reward -= 25;

  return reward;
}

void RLLogger::fill_state(const MPCCore& mpc_core, mpcc::RLState& state) {
  Eigen::VectorXd mpc_state       = mpc_core.get_state();
  std::array<double, 2> mpc_input = mpc_core.get_mpc_command();
  bool solver_status              = mpc_core.get_solver_status();

  Eigen::VectorXd cbf_data_abv = mpc_core.get_cbf_data(
      mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
  Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
      mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

  double alpha_abv = mpc_core.get_params().at("CBF_ALPHA_ABV");
  double alpha_blw = mpc_core.get_params().at("CBF_ALPHA_BLW");

  double max_vel       = mpc_core.get_params().at("LINVEL");
  double curr_progress = mpc_state[5] / max_vel;

  std::array<Eigen::VectorXd, 2> state_limits = mpc_core.get_state_limits();
  std::array<Eigen::VectorXd, 2> input_limits = mpc_core.get_input_limits();

  state.state.resize(12);

  state.state[0] =
      normalize(mpc_state[2], state_limits[0][2], state_limits[1][2]);
  state.state[1] =
      normalize(mpc_state[3], state_limits[0][3], state_limits[1][3]);
  state.state[2] =
      normalize(mpc_input[0], input_limits[0][0], input_limits[1][0]);
  state.state[3] =
      normalize(mpc_input[1], input_limits[0][1], input_limits[1][1]);
  state.state[4] = normalize(cbf_data_abv[1], 0, _max_obs_dist);
  state.state[5] = normalize(cbf_data_blw[1], 0, _max_obs_dist);
  state.state[6] = normalize(cbf_data_abv[2], -M_PI, M_PI);

  // curr progress is already normalized, can't norm h value
  state.state[7]      = curr_progress;
  state.state[8]      = cbf_data_abv[0];
  state.state[9]      = cbf_data_blw[0];
  state.state[10]     = alpha_abv;
  state.state[11]     = alpha_blw;
  state.solver_status = solver_status;
}

std::string RLLogger::serialize_state(const mpcc::RLState& state) {
  uint32_t serial_size =
      ros::serialization::serializationLength(_prev_rl_state);
  std::vector<uint8_t> buffer(serial_size);
  ros::serialization::OStream stream(buffer.data(), serial_size);
  ros::serialization::serialize(stream, state);

  return std::string(buffer.begin(), buffer.end());
}

}  // namespace logger
