#pragma once

#include <map>
#include <vector>
#include <nlopt.hpp>
#include <Eigen/Dense>

#include <uav_mpc/types.h>

// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_unicycle_model_mpcc.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#define NX UNICYCLE_MODEL_MPCC_NX
#define NP UNICYCLE_MODEL_MPCC_NP
#define NU UNICYCLE_MODEL_MPCC_NU
#define NBX0 UNICYCLE_MODEL_MPCC_NBX0

class MPCC
{
public:
    MPCC();
    ~MPCC();

    std::vector<double> solve(const Eigen::VectorXd &state);
    void set_tubes(const std::vector<SplineWrapper> &tubes);
    void load_params(const std::map<std::string, double> &params);
    void set_reference(const std::vector<Spline1D> &reference, double arclen);

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

    double get_s_from_state(const Eigen::VectorXd &state);

    std::map<std::string, double> _params;
    std::vector<double> _prev_x0;
    std::vector<Spline1D> _reference;

    Eigen::VectorXd _state;

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
    int _ind_inc;

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

    double _w_angvel;
    double _w_angvel_d;
    double _w_linvel_d;

    unsigned int iterations;

    bool _use_cbf;
    bool _use_eigen;

    double *_new_time_steps;

    unicycle_model_mpcc_solver_capsule *_acados_ocp_capsule;
    ocp_nlp_in *_nlp_in;
    ocp_nlp_out *_nlp_out;
    ocp_nlp_dims *_nlp_dims;
    ocp_nlp_config *_nlp_config;
    ocp_nlp_solver *_nlp_solver;
    void *_nlp_opts;
};
