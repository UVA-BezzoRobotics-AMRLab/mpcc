#pragma once

#include <map>
#include <vector>
#include <nlopt.hpp>
#include <Eigen/Dense>

#include <uav_mpc/types.h>

// backward
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

// forward
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

struct Segment
{
    double s0, s1;
    double mx, bx;
    double my, by;
    double dmx, dbx;
    double dmy, dby;
}; typedef struct Segment Segment_t;

class MPCC
{
public:
    MPCC();
    ~MPCC();

    std::vector<double> solve(const Eigen::VectorXd &state);
    void load_params(const std::map<std::string, double> &params);
    void set_reference(const std::vector<Spline1D> &reference, double arclen);
    void set_reference(const std::vector<SplineWrapper> &reference, double arclen);
    void set_segments(const std::vector<Segment_t> &segments);
    void set_tubes(const std::vector<SplineWrapper> &tubes);

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

    static double objective(const std::vector<double> &x, std::vector<double> &grad, void *data);
    static double constraint(const std::vector<double> &x, std::vector<double> &grad, void *data);

    static void multi_constraint(unsigned m, double *result, unsigned n, const double *x, double *grad, void *f_data);

    // objective
    static autodiff::real eval_objective(const autodiff::ArrayXreal& x, void *data);

    // constraints
    // static double eval_h_func(const std::vector<double> &x, void *data);
    // static std::vector<double> eval_cbf_constraint(const std::vector<double> &vars, void *data);
    static autodiff::VectorXreal eval_cbf_constraint(const autodiff::VectorXreal &vars, void *data);
    static autodiff::VectorXreal eval_constraint(const autodiff::VectorXreal &x, void *data);

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

    std::map<std::string, double> _params;

    std::vector<double> prev_x0;
    std::vector<Spline1D> _reference;
    std::vector<Segment_t> _segments;
    std::vector<SplineWrapper> _reference_tk;
    std::vector<SplineWrapper> _tubes;

    Eigen::VectorXd _state;

};
