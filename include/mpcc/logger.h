#pragma once

#include <amrl_logging/LoggingBufferCheck.h>
#include <amrl_logging/LoggingData.h>
#include <amrl_logging/LoggingDropTable.h>
#include <amrl_logging/LoggingStart.h>
#include <amrl_logging/LoggingStop.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <amrl_logging_util/util.hpp>

namespace logger
{

struct logger_state
{
    double theta;
    double vel;
    double acc;
    double ang_vel;
    double obs_dist;
    double obs_heading;
    double progress;
    double h_val;
    double alpha_val;
};
typedef struct logger_state logger_state_t;

class Logger
{
   public:
    Logger(ros::NodeHandle& nh);

    ~Logger();

    void log_row(const logger_state_t& prev_state, const logger_state_t& curr_state,
                 double alpha_dot, double reward);

   private:
    void collision_cb(const std_msgs::Bool::ConstPtr& msg);

    ros::NodeHandle _nh;

    ros::Publisher _logging_pub;

    ros::Subscriber _collision_sub;

    logger_state_t _prev_state;
    logger_state_t _curr_state;

    unsigned int _count;

    std::string _table_name;
    std::string _topic_name;

    bool _is_done;
};

}  // namespace logger
