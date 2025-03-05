#include <mpcc/logger.h>

namespace logger
{

Logger::Logger(ros::NodeHandle& nh)
{
    _table_name = "replay_buffer";
    _topic_name = "/cbf_rl_learning";

    _logging_pub = _nh.advertise<amrl_logging::LoggingData>(_topic_name, 100);

    _collision_sub = _nh.subscribe("/collision", 1, &Logger::collision_cb, this);

    _count   = 0;
    _is_done = false;

    const std::vector<std::string> string_types({"is_done"});
    const std::vector<std::string> float_types({"id",
                                                "prev_theta",
                                                "prev_vel",
                                                "prev_acc",
                                                "prev_angvel",
                                                "prev_obs_dist",
                                                "prev_obs_heading",
                                                "prev_progress",
                                                "prev_h",
                                                "prev_alpha",
                                                "action",
                                                "reward",
                                                "curr_theta",
                                                "curr_vel",
                                                "curr_acc",
                                                "curr_angvel",
                                                "curr_obs_dist",
                                                "curr_obs_heading",
                                                "curr_progress",
                                                "curr_h",
                                                "curr_alpha"});

    if (!amrl::logging_setup(_nh, _table_name, _topic_name, string_types, float_types))
    {
        ROS_ERROR("[Logger] Failed to setup logging");
        exit(-1);
    }
}

void Logger::collision_cb(const std_msgs::Bool::ConstPtr& msg) {}

Logger::~Logger() { amrl::logging_finish(_nh, _table_name); }

void Logger::log_row(const logger_state_t& prev_state, const logger_state_t& curr_state,
                     double alpha_dot, double reward)
{
    amrl_logging::LoggingData row;
    std::string is_done_str              = _is_done ? "true" : "false";
    std::vector<std::string> string_data = {"true"};
    std::vector<double> numeric_data     = {
        (double)_count,
        prev_state.theta,        // theta
        prev_state.vel,          // velocity
        prev_state.acc,          // acceleration
        prev_state.ang_vel,      // angular velocity
        prev_state.obs_dist,     // distance to obstacle
        prev_state.obs_heading,  // heading to obstacle
        prev_state.progress,     // progress
        prev_state.h_val,        // h value
        prev_state.alpha_val,    // alpha value
        alpha_dot,
        reward,
        prev_state.theta,        // theta
        prev_state.vel,          // velocity
        prev_state.acc,          // acceleration
        prev_state.ang_vel,      // angular velocity
        prev_state.obs_dist,     // distance to obstacle
        prev_state.obs_heading,  // heading to obstacle
        prev_state.progress,     // progress
        prev_state.alpha_val     // alpha value
    };
    row.header.seq += _count++;
    row.header.stamp = ros::Time::now();
    row.labels       = string_data;
    row.values       = numeric_data;

    _logging_pub.publish(row);
}

}  // namespace logger
