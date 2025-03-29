#pragma once

#include <mpcc/mpcc_acados.h>
#include <mpcc/types.h>

#include <map>
#include <memory>

class MPCCore
{
   public:
    MPCCore();

    ~MPCCore();

    void load_params(const std::map<std::string, double> &params);
    /**********************************************************************
     * Function: MPCCore::load_params()
     * Description: Loads parameters for the MPC controller
     * Parameters:
     * @param params: const std::map<std::string, double>&
     * Returns:
     * N/A
     * Notes:
     * This function loads parameters for the MPC controller, including
     * the time step, maximum angular acc, linear acc, linear vel, angular
     * vel, and prop controller gains and thresholds
     **********************************************************************/

    bool orient_robot();
    /**********************************************************************
     * Function: MPCCore::orient_robot()
     * Description: Orients the robot to the reference trajectory
     * Parameters:
     * N/A
     * Returns:
     * true if robot is not aligned with reference. False otherwise.
     * Notes:
     * A proportional controller is used to align the robot with the
     * reference trajectory if the error is larger than the threshold
     **********************************************************************/

    std::array<double, 2> solve(const Eigen::VectorXd &state);
    /**********************************************************************
     * Function: MPCCore::solve()
     * Description: Solves the MPC problem for the current timestep
     * Parameters:
     * @param state: const Eigen::VectorXd&
     * Returns:
     * control input for the next timestep
     * Notes:
     * This function solves the MPC problem and returns the control
     * inputs for the robot. If the robot's heading is not aligned with
     * the reference trajectory, orient_robot generates the control inputs
     **********************************************************************/

    /***********************
     * Setters and Getters
     ***********************/
    void set_state(const Eigen::Vector3d &state);
    void set_odom(const Eigen::Vector3d &odom);
    void set_goal(const Eigen::Vector2d &goal);
    void set_trajectory(const std::array<Spline1D, 2> &ref, double arclen);
    void set_tubes(const std::array<Eigen::VectorXd, 2> &tubes);

    const bool get_solver_status() const;
    const Eigen::VectorXd &get_state() const;
    std::vector<Eigen::VectorXd> get_horizon() const;
    const std::array<double, 2> &get_mpc_results() const;
    const std::map<std::string, double> &get_params() const;
    Eigen::VectorXd get_cbf_data(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                                 bool is_abv) const;

   protected:
    double limit(double prev_v, double input, double max_rate) const;
    /**********************************************************************
     * Function: MPCCore::limit()
     * Description: Limits the input to the maximum rate
     * Parameters:
     * @param prev_v: double
     * @param input: double
     * @param max_rate: double
     * Returns:
     * rate limited input
     * Notes:
     * This function limits the input to the maximum rate
     **********************************************************************/

    double _dt;
    double _max_anga;
    double _max_linacc;
    double _curr_vel;
    double _curr_angvel;
    double _max_vel;
    double _max_angvel;
    double _ref_length;

    double _prop_gain;
    double _prop_angle_thresh;

    bool _is_set;
    bool _use_cbf;
    bool _traj_reset;

    std::array<double, 2> _mpc_results;
    std::vector<traj_point_t> _trajectory;

    std::array<Spline1D, 2> _ref;

    Eigen::Vector3d _odom;
    Eigen::Vector2d _goal;

    // learning states
    Eigen::VectorXd _prev_rl_state;
    Eigen::VectorXd _curr_rl_state;

    std::map<std::string, double> _params;

    std::unique_ptr<MPCC> _mpc;
};
