#pragma once

#include <amrl_logging/LoggingBufferCheck.h>
#include <amrl_logging/LoggingData.h>
#include <amrl_logging/LoggingDropTable.h>
#include <amrl_logging/LoggingStart.h>
#include <amrl_logging/LoggingStop.h>
#include <mpcc/QuerySAC.h>
#include <mpcc/QuerySACDI.h>
#include <mpcc/mpcc_core.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <amrl_logging_util/util.hpp>
#include <cstdint>

namespace logger {

struct logger_state {
  double theta;
  double vel;
  double acc;
  double ang_vel;
  double obs_dist_abv;
  double obs_dist_blw;
  double obs_heading;
  double progress;
  double h_val_abv;
  double h_val_blw;
  double alpha_val_abv;
  double alpha_val_blw;
  bool solver_status;
};
typedef struct logger_state logger_state_t;

struct logger_state_di {
  double vx;
  double vy;
  double ax;
  double ay;
  double obs_dist_abv;
  double obs_dist_blw;
  double obs_heading;
  double progress;
  double h_val_abv;
  double h_val_blw;
  double alpha_val_abv;
  double alpha_val_blw;
  bool solver_status;
};
typedef struct logger_state_di logger_state_di_t;

class RLLogger {
 public:
  RLLogger(ros::NodeHandle& nh, double min_alpha, double max_alpha,
           bool is_logging, const std::string& mpc_type);

  ~RLLogger();

  void log_transition(const MPCCore& mpc_core, double len_start,
                      double ref_len);
  bool request_alpha(MPCCore& mpc_core, double ref_len);

 private:
  void collision_cb(const std_msgs::Bool::ConstPtr& msg);

  double compute_reward();
  double compute_reward_di();

  ros::NodeHandle _nh;

  ros::Publisher _done_pub;
  ros::Publisher _logging_pub;
  ros::Publisher _alpha_pub_abv;
  ros::Publisher _alpha_pub_blw;

  ros::Subscriber _collision_sub;

  ros::ServiceClient _sac_srv;

  logger_state_t _prev_rl_state;
  logger_state_t _curr_rl_state;

  logger_state_di_t _prev_rl_state_di;
  logger_state_di_t _curr_rl_state_di;

  unsigned int _count;

  std::string _table_name;
  std::string _topic_name;
  std::string _mpc_type;

  double _min_alpha;
  double _max_alpha;
  double _alpha_dot_abv;
  double _alpha_dot_blw;

  bool _is_done;
  bool _is_logging;
  bool _is_colliding;
  bool _is_first_iter;

  uint8_t _exceeded_bounds;
};

}  // namespace logger
