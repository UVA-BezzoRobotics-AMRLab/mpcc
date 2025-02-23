#pragma once

#include <mpcc/types.h>

#include <Eigen/Dense>
#include <map>
#include <nlopt.hpp>
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

    void set_odom(const Eigen::VectorXd &odom);
    void set_tubes(const std::array<Eigen::VectorXd, 2> &tubes);
    void load_params(const std::map<std::string, double> &params);
    void set_reference(const std::array<Spline1D, 2> &reference, double arclen);

    void reset_horizon();

    Eigen::VectorXd get_state();

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
    std::vector<Spline1D> get_ref_from_s(double s);
    double get_s_from_state(const Eigen::VectorXd &state);
    Eigen::VectorXd next_state(const Eigen::VectorXd &current_state,
                               const Eigen::VectorXd &control);

    void warm_start_no_u(double *x_init);
    void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd &state);
    void process_solver_output(double s);

    bool set_solver_parameters(const std::vector<Spline1D> &ref);

    std::map<std::string, double> _params;

    Eigen::VectorXd _prev_x0;
    Eigen::VectorXd _prev_u0;

    std::array<Spline1D, 2> _reference;
    std::array<Eigen::VectorXd, 2> _tubes;

    Eigen::VectorXd _state;
    Eigen::VectorXd _odom;

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

    double _alpha;
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
