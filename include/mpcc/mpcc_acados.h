#pragma once

#include <mpcc/types.h>

#include <Eigen/Dense>
#include <map>
#include <vector>

// acados
#include "acados_c/ocp_nlp_interface.h"
#include "acados_sim_solver_unicycle_model_mpcc.h"
#include "acados_solver_unicycle_model_mpcc.h"

#define NX UNICYCLE_MODEL_MPCC_NX
#ifdef UNICYCLE_MODEL_MPCC_NS
#define NS UNICYCLE_MODEL_MPCC_NS
#endif
#define NP UNICYCLE_MODEL_MPCC_NP
#define NU UNICYCLE_MODEL_MPCC_NU
#define NBX0 UNICYCLE_MODEL_MPCC_NBX0

class MPCC
{
   public:
    MPCC();
    ~MPCC();

    std::array<double, 2> solve(const Eigen::VectorXd &state);

    void load_params(const std::map<std::string, double> &params);
    /**********************************************************************
     * Function: MPCC::load_params()
     * Description: Loads parameters for the MPC controller
     * Parameters:
     * @param params: const std::map<std::string, double>&
     * Returns:
     * N/A
     * Notes:
     * This function loads parameters for the MPC controller, including
     * the time step, maximum angular and body rates, weights for the
     * mpcc cost function, and CBF/CLF parameters
     **********************************************************************/

    /***********************
     * Setters and Getters
     ***********************/
    void reset_horizon();
    void set_odom(const Eigen::VectorXd &odom);
    void set_tubes(const std::array<Eigen::VectorXd, 2> &tubes);
    void set_reference(const std::array<Spline1D, 2> &reference, double arclen);
    void set_dyna_obs(const Eigen::MatrixXd &dyna_obs);

    const Eigen::VectorXd &get_state() const;
    const bool get_solver_status() const;
    Eigen::VectorXd get_cbf_data(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                                 bool is_abv) const;

    // TOOD: make getter for these
    std::vector<double> mpc_x;
    std::vector<double> mpc_y;
    std::vector<double> mpc_theta;
    std::vector<double> mpc_linvels;
    std::vector<double> mpc_s;
    std::vector<double> mpc_s_dot;

    std::vector<double> mpc_angvels;
    std::vector<double> mpc_linaccs;
    std::vector<double> mpc_s_ddots;

   protected:
    std::array<Spline1D, 2> compute_adjusted_ref(double s) const;
    /**********************************************************************
     * Function: MPCC::get_ref_from_s()
     * Description: Generates a reference trajectory from a given arc length
     * Parameters:
     * @param s: double
     * Returns:
     * a reparameterized trajectory starting at arc length s=0
     * Notes:
     * If the trajectory is shorter than required mpc size, then the
     * last point is repeated for spline generation.
     **********************************************************************/

    double get_s_from_state(const Eigen::VectorXd &state);
    /**********************************************************************
     * Function: MPCC::get_s_from_state()
     * Description: Get the arc length of closest point on reference trajectory
     * Parameters:
     * @param state: const Eigen::VectorXd&
     * Returns:
     * arc length value of closest point to state
     **********************************************************************/

    Eigen::VectorXd next_state(const Eigen::VectorXd &current_state,
                               const Eigen::VectorXd &control);
    /**********************************************************************
     * Function: MPCC::next_state()
     * Description: Calculates the next state of the robot given current
     * state and control input
     * Parameters:
     * @param current_state: const Eigen::VectorXd&
     * @param control: const Eigen::VectorXd&
     * Returns:
     * Next state of the robot
     **********************************************************************/

    void warm_start_no_u(double *x_init);
    /**********************************************************************
     * Function: MPCC::warm_start_no_u()
     * Description: Warm starts the MPC solver with no control inputs
     * Parameters:
     * @param x_init: double*
     * Returns:
     * N/A
     * Notes:
     * This function sets the initial state for the MPC solver assuming
     * a 0 control input
     **********************************************************************/

    void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd &state);
    /**********************************************************************
     * Function: MPCC::warm_start_shifted_u()
     * Description: Warm starts the MPC solver with shifted control inputs
     * Parameters:
     * @param correct_perturb: bool
     * @param state: const Eigen::VectorXd&
     * Returns:
     * N/A
     * Notes:
     * This function sets the initial state for the MPC solver by shifting
     * the control inputs and states from the previous solution.
     * See From linear to nonlinear MPC: bridging the gap via the
     * real-time iteration, Gros et. al. for more details.
     **********************************************************************/

    void process_solver_output(double s);
    /**********************************************************************
     * Function: MPCC::process_solver_output()
     * Description: Processes the output of the MPC solver
     * Parameters:
     * @param s: double
     * Returns:
     * N/A
     **********************************************************************/

    bool set_solver_parameters(const std::array<Spline1D, 2> &adjusted_ref);
    /**********************************************************************
     * Function: MPCC::set_solver_parameters()
     * Description: Sets the parameters for the MPC solver
     * Parameters:
     * @param ref: const std::vector<Spline1D>&
     * Returns:
     * bool - true if successful, false otherwise
     **********************************************************************/

    void apply_affine_transform(Eigen::VectorXd &state, const Eigen::Vector2d &rot_point,
                                const Eigen::MatrixXd &m_affine);
    /**********************************************************************
     * Function: MPCC::apply_affine_transform()
     * Description: Applies an affine transformation to state in place
     * Parameters:
     * @param state: const Eigen::VectorXd&
     * @param rot_point: const Eigen::Vector2d&
     * @param m_affine: const Eigen::MatrixXd&
     * Returns:
     * Eigen::VectorXd - Transformed state
     * Notes:
     * Applies affine transformation defined my m_affine to state,
     * rotation occurs about rot_point
     ***********************************************************************/

    std::map<std::string, double> _params;

    Eigen::VectorXd _prev_x0;
    Eigen::VectorXd _prev_u0;

    std::array<Spline1D, 2> _reference;
    std::array<Eigen::VectorXd, 2> _tubes;

    Eigen::VectorXd _state;
    Eigen::VectorXd _odom;
    Eigen::MatrixXd _dyna_obs;

    int _mpc_steps;
    int _x_start;
    int _y_start;
    int _theta_start;
    int _v_start;
    int _s_start;
    int _s_dot_start;
    int _angvel_start;
    int _linacc_start;
    int _s_ddot_start;
    int _ind_state_inc;
    int _ind_input_inc;

    int _ref_samples;

    double _dt;
    double _ds;

    double _bound_value;
    double _max_linvel;
    double _max_angvel;
    double _max_linacc;

    double _alpha_abv;
    double _alpha_blw;
    double _colinear;
    double _padding;

    double _s_dot;
    double _ref_length;

    double _gamma;
    double _w_ql_lyap;
    double _w_qc_lyap;

    double _w_angvel;
    double _w_angvel_d;
    double _w_linvel_d;
    double _w_ql;
    double _w_qc;
    double _w_q_speed;

    double _ref_len_sz;

    unsigned int iterations;

    bool _use_cbf;
    bool _use_eigen;
    bool _is_shift_warm;
    bool _solve_success;
    bool _use_dyna_obs;

    double *_new_time_steps;

    unicycle_model_mpcc_sim_solver_capsule *_acados_sim_capsule;
    unicycle_model_mpcc_solver_capsule *_acados_ocp_capsule;

    sim_config *_sim_config;
    sim_in *_sim_in;
    sim_out *_sim_out;
    void *_sim_dims;

    ocp_nlp_in *_nlp_in;
    ocp_nlp_out *_nlp_out;
    ocp_nlp_dims *_nlp_dims;
    ocp_nlp_config *_nlp_config;
    ocp_nlp_solver *_nlp_solver;
    void *_nlp_opts;
};
