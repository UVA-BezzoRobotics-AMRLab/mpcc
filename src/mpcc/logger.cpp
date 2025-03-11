#include <mpcc/logger.h>
#include <std_msgs/Float64.h>

#include <Eigen/Core>
#include <string>

namespace logger
{

RLLogger::RLLogger(ros::NodeHandle& nh, double min_alpha, double max_alpha)
{
    _min_alpha = min_alpha;
    _max_alpha = max_alpha;

    _table_name = "replay_buffer";
    _topic_name = "/cbf_rl_learning";

    _done_pub    = _nh.advertise<std_msgs::Bool>("/mpc_done", 100);
    _alpha_pub   = _nh.advertise<std_msgs::Float64>("/cbf_alpha", 100);
    _logging_pub = _nh.advertise<amrl_logging::LoggingData>(_topic_name, 100);

    _collision_sub = _nh.subscribe("/collision", 1, &RLLogger::collision_cb, this);

    _sac_srv = nh.serviceClient<mpcc::QuerySAC>("/query_sac");

    _count           = 0;
    _is_done         = false;
    _is_first_iter   = true;
    _exceeded_bounds = false;
    _is_colliding    = false;

    _alpha_dot = 0.;

    const std::vector<std::string> string_types({"is_done"});
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
                                                "prev_alpha",
                                                "action",
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
                                                "curr_alpha"});

    if (!amrl::logging_setup(_nh, _table_name, _topic_name, string_types, {}, float_types))
    {
        ROS_ERROR("[Logger] Failed to setup logging");
        exit(-1);
    }
}

RLLogger::~RLLogger() { amrl::logging_finish(_nh, _table_name); }

void RLLogger::collision_cb(const std_msgs::Bool::ConstPtr& msg) { _is_colliding = msg->data; }

bool RLLogger::request_alpha(MPCCore& mpc_core)
{
    Eigen::VectorXd mpc_state       = mpc_core.get_state();
    std::array<double, 2> mpc_input = mpc_core.get_mpc_results();

    Eigen::VectorXd cbf_data_abv =
        mpc_core.get_cbf_data(mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
    Eigen::VectorXd cbf_data_blw =
        mpc_core.get_cbf_data(mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

    double alpha = mpc_core.get_params().at("CBF_ALPHA");

    mpcc::QuerySAC req;
    req.request.theta        = mpc_state[2];
    req.request.vel          = mpc_state[3];
    req.request.acc          = mpc_input[1];
    req.request.ang_vel      = mpc_input[0];
    req.request.obs_dist_abv = cbf_data_abv[1];
    req.request.obs_dist_blw = cbf_data_blw[1];
    req.request.heading_dist = cbf_data_abv[2];
    req.request.progress     = mpc_state[4];
    req.request.h_val_abv    = cbf_data_abv[0];
    req.request.h_val_blw    = cbf_data_blw[0];
    req.request.alpha        = mpc_core.get_params().at("CBF_ALPHA");

    if (_sac_srv.call(req))
    {
        if (!req.response.success) ROS_ERROR("SAC service failed");
        // integrate alpha_dot into CBF_ALPHA
        // clip alpha to ensure it's within bounds
        std::map<std::string, double> mpc_params = mpc_core.get_params();
        double dt                                = mpc_params.at("DT");

        _alpha_dot   = req.response.alpha_dot;
        double alpha = mpc_params["CBF_ALPHA"] + _alpha_dot * dt;

        if (alpha < _min_alpha || alpha > _max_alpha) _exceeded_bounds = true;

        alpha = std::max(_min_alpha, std::min(_max_alpha, alpha));

        std_msgs::Float64 alpha_msg;
        alpha_msg.data = alpha;
        _alpha_pub.publish(alpha_msg);

        mpc_params["CBF_ALPHA"] = alpha;
        mpc_core.load_params(mpc_params);
    }
    else
    {
        ROS_ERROR("Failed to call service query_sac");
        return false;
    }

    return true;
}

void RLLogger::log_transition(const MPCCore& mpc_core, double len_start, double ref_len)
{
    if (_is_first_iter)
    {
        Eigen::VectorXd mpc_state       = mpc_core.get_state();
        std::array<double, 2> mpc_input = mpc_core.get_mpc_results();

        Eigen::VectorXd cbf_data_abv =
            mpc_core.get_cbf_data(mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
        Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
            mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

        double alpha = mpc_core.get_params().at("CBF_ALPHA");

        _curr_rl_state.theta        = mpc_state[2];
        _curr_rl_state.vel          = mpc_state[3];
        _curr_rl_state.obs_dist_abv = cbf_data_abv[1];
        _curr_rl_state.obs_dist_blw = cbf_data_blw[1];
        _curr_rl_state.obs_heading  = cbf_data_abv[2];
        _curr_rl_state.progress     = mpc_state[4];
        _curr_rl_state.h_val_abv    = cbf_data_abv[0];
        _curr_rl_state.h_val_blw    = cbf_data_blw[0];
        _curr_rl_state.alpha_val    = alpha;
        _curr_rl_state.ang_vel      = mpc_input[0];
        _curr_rl_state.acc          = mpc_input[1];

        _prev_rl_state = _curr_rl_state;

        _is_first_iter = false;
    }
    // we don't want to log if already reported an is_done state
    else if (!_is_done)
    {
        double reward = -100;

        const Eigen::VectorXd& mpc_state = mpc_core.get_state();
        std::array<double, 2> mpc_input  = mpc_core.get_mpc_results();

        Eigen::VectorXd cbf_data_abv =
            mpc_core.get_cbf_data(mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), true);
        Eigen::VectorXd cbf_data_blw = mpc_core.get_cbf_data(
            mpc_state, Eigen::Vector2d(mpc_input[0], mpc_input[1]), false);

        double alpha = mpc_core.get_params().at("CBF_ALPHA");

        _curr_rl_state.theta        = mpc_state[2];
        _curr_rl_state.vel          = mpc_state[3];
        _curr_rl_state.obs_dist_abv = cbf_data_abv[1];
        _curr_rl_state.obs_dist_blw = cbf_data_blw[1];
        _curr_rl_state.obs_heading  = cbf_data_abv[2];
        _curr_rl_state.progress     = mpc_state[4];
        _curr_rl_state.h_val_abv    = cbf_data_abv[0];
        _curr_rl_state.h_val_blw    = cbf_data_blw[0];
        _curr_rl_state.alpha_val    = alpha;
        _curr_rl_state.ang_vel      = mpc_input[0];
        _curr_rl_state.acc          = mpc_input[1];

        if (!_is_colliding)
        {
            // weight distance to obstacle
            // reward = 5 * _curr_rl_state.obs_dist_abv * _curr_rl_state.obs_dist_blw;
        }
        else
        {
            _is_done = true;
        }

        // add penalty for not making progress
        // reward -= 12 * (1 - _curr_rl_state(6));
        // ref len should never be negative, but fabs just in case
        // need to use passed in len-start because core version is relative
        if (fabs(ref_len) > 1e-3) reward -= 12 * (ref_len - len_start) / ref_len;

        // add small penalty for large alpha jumps
        reward -= 0.1 * _alpha_dot * _alpha_dot;

        // add penalty for using higher alpha values
        // reward -= .1 * (_curr_rl_state(8)- _min_alpha);

        // if alpha value is outside bounds, penalize heavily
        if (_exceeded_bounds) reward -= 20;

        _exceeded_bounds = false;

        // if h_value is negative, penalize heavily
        if (_curr_rl_state.h_val_abv < 0) reward -= 10;
        if (_curr_rl_state.h_val_blw < 0) reward -= 10;

        // log to database
        amrl_logging::LoggingData row;
        std::string is_done_str              = _is_done ? "true" : "false";
        std::vector<std::string> string_data = {is_done_str};
        std::vector<double> numeric_data     = {
            _count,
            _prev_rl_state.theta,         // theta
            _prev_rl_state.vel,           // velocity
            _prev_rl_state.acc,           // acceleration
            _prev_rl_state.ang_vel,       // angular velocity
            _prev_rl_state.obs_dist_abv,  // distance to obstacle abv
            _prev_rl_state.obs_dist_blw,  // distance to obstacle blw
            _prev_rl_state.obs_heading,   // heading to obstacle
            _prev_rl_state.progress,      // progress
            _prev_rl_state.h_val_abv,     // h value
            _prev_rl_state.h_val_blw,     // h value
            _prev_rl_state.alpha_val,     // alpha value
            _alpha_dot,
            reward,
            _curr_rl_state.theta,         // theta
            _curr_rl_state.vel,           // velocity
            _curr_rl_state.acc,           // acceleration
            _curr_rl_state.ang_vel,       // angular velocity
            _curr_rl_state.obs_dist_abv,  // distance to obstacle abv
            _curr_rl_state.obs_dist_blw,  // distance to osbtacle blw
            _curr_rl_state.obs_heading,   // heading to obstacle
            _curr_rl_state.progress,      // progress
            _curr_rl_state.h_val_abv,     // h value
            _curr_rl_state.h_val_blw,     // h value
            _curr_rl_state.alpha_val};    // alpha value

        row.header.seq += _count++;
        row.header.stamp = ros::Time::now();
        row.labels       = string_data;
        row.reals        = numeric_data;

        _logging_pub.publish(row);

        _prev_rl_state = _curr_rl_state;
    }

    if (_is_done)
    {
        std_msgs::Bool done_msg;
        done_msg.data = true;
        _done_pub.publish(done_msg);
    }
}

}  // namespace logger
