#pragma once

#include <mpcc/QuerySAC.h>
#include <mpcc/QuerySACDI.h>
#include <mpcc/RLState.h>
#include <mpcc/mpcc_core.h>

#include <amrl_logging/LoggingBufferCheck.h>
#include <amrl_logging/LoggingData.h>
#include <amrl_logging/LoggingDropTable.h>
#include <amrl_logging/LoggingStart.h>
#include <amrl_logging/LoggingStop.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <Eigen/Core>
#include <amrl_logging_util/util.hpp>

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace logger {

class RLLogger {
 public:
  RLLogger(ros::NodeHandle& nh,
           const std::unordered_map<std::string_view, double>& params,
           bool is_logging, const std::string& mpc_type);

  void load_params(const std::unordered_map<std::string_view, double>& params);

  ~RLLogger();

  void log_transition(const MPCCore& mpc_core, double len_start,
                      double ref_len);
  bool request_alpha(MPCCore& mpc_core, double ref_len);

 private:
  void collision_cb(const std_msgs::Bool::ConstPtr& msg);
  void fill_state(const MPCCore& mpc_core, mpcc::RLState& state);

  std::string serialize_state(const mpcc::RLState& state);

  double compute_reward();

  ros::NodeHandle _nh;

  ros::Publisher _done_pub;
  ros::Publisher _logging_pub;
  ros::Publisher _alpha_pub_abv;
  ros::Publisher _alpha_pub_blw;

  ros::Subscriber _collision_sub;

  ros::ServiceClient _sac_srv;

  mpcc::RLState _prev_rl_state;
  mpcc::RLState _curr_rl_state;

  unsigned int _count;

  std::string _table_name;
  std::string _topic_name;
  std::string _mpc_type;

  double _min_alpha;
  double _max_alpha;
  double _min_alpha_dot;
  double _max_alpha_dot;
  double _min_h_val;
  double _max_h_val;

  double _max_obs_dist;

  double _alpha_dot_abv;
  double _alpha_dot_blw;

  bool _is_done;
  bool _is_logging;
  bool _is_colliding;
  bool _is_first_iter;

  uint8_t _exceeded_bounds;

  std::unordered_map<std::string_view, double> _params;
};

double normalize(double val, double min, double max);

}  // namespace logger
