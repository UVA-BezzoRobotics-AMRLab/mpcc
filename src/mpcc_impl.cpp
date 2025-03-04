#include <string>
#include <math.h>
#include <iostream>
#include <ros/ros.h>
#include <cppad/ipopt/solve.hpp>

#include <uav_mpc/mpcc_impl.h>

// ROS IPOPT-based MPC for differential drive/skid-steering vehicles
// Code modifies the car-based MPC tracking github below:
// https://github.com/Danziger/CarND-T2-P5-Model-Predictive-Control-MPC

namespace
{
    using CppAD::AD;

    class FG_eval
    {
    public:
        double _dt, _ref_cte, _ref_etheta, _ref_vel;
        double _w_cte, _w_etheta, _w_vel, _w_angvel, _w_angvel_d, _w_linvel_d, _w_pos;
        int _mpc_steps;

        bool _use_eigen = false;

        AD<double> cost_cte, cost_etheta, cost_vel;

        std::vector<Spline1D> _reference;
        std::vector<SplineWrapper> _reference_tk;
        std::vector<SplineWrapper> _tubes;

        FG_eval()
        {

            // Set default value
            _dt = 0.1; // in sec
            _ref_cte = 0;
            _ref_etheta = 0;
            _ref_vel = 0.5; // m/s

            _use_cbf = false;
            _alpha = 1.0;
            _colinear = 0.01;
            _padding = .05;

            // Cost function weights
            _w_cte = 100;
            _w_etheta = .5;
            _w_vel = 0;
            _w_angvel = 0;
            _w_angvel_d = 0;
            _w_linvel_d = 0;
            _w_pos = 1;

            _mpc_steps = 10;
            _x_start = 0;
            _y_start = _x_start + _mpc_steps;
            _theta_start = _y_start + _mpc_steps;
            _v_start = _theta_start + _mpc_steps;
            _s_start = _v_start + _mpc_steps;
            _s_dot_start = _s_start + _mpc_steps;
            _angvel_start = _s_dot_start + _mpc_steps;
            _linacc_start = _angvel_start + _mpc_steps - 1;
            _s_ddot_start = _linacc_start + _mpc_steps - 1;
        }

        // Load parameters for constraints
        void load_params(const std::map<std::string, double> &params)
        {
            _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
            _mpc_steps = params.find("STEPS") != params.end() ? params.at("STEPS") : _mpc_steps;
            _ref_cte = params.find("REF_CTE") != params.end() ? params.at("REF_CTE") : _ref_cte;
            _ref_etheta = params.find("REF_ETHETA") != params.end() ? params.at("REF_ETHETA") : _ref_etheta;
            _ref_vel = params.find("REF_V") != params.end() ? params.at("REF_V") : _ref_vel;

            _w_pos = params.find("W_POS") != params.end() ? params.at("W_POS") : _w_pos;
            _w_cte = params.find("W_CTE") != params.end() ? params.at("W_CTE") : _w_cte;
            _w_etheta = params.find("W_ETHETA") != params.end() ? params.at("W_ETHETA") : _w_etheta;
            _w_vel = params.find("W_V") != params.end() ? params.at("W_V") : _w_vel;
            _w_angvel = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL") : _w_angvel;
            _w_angvel_d = params.find("W_DANGVEL") != params.end() ? params.at("W_DANGVEL") : _w_angvel_d;
            _w_linvel_d = params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

            _use_cbf = params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
            _alpha = params.find("CBF_ALPHA") != params.end() ? params.at("CBF_ALPHA") : _alpha;
            _colinear = params.find("CBF_COLINEAR") != params.end() ? params.at("CBF_COLINEAR") : _colinear;
            _padding = params.find("CBF_PADDING") != params.end() ? params.at("CBF_PADDING") : _padding;

            // print all parameters
            // std::cout << "DT: " << _dt << std::endl;
            // std::cout << "STEPS: " << _mpc_steps << std::endl;
            // std::cout << "REF_CTE: " << _ref_cte << std::endl;
            // std::cout << "REF_ETHETA: " << _ref_etheta << std::endl;
            // std::cout << "REF_V: " << _ref_vel << std::endl;
            // std::cout << "W_POS: " << _w_pos << std::endl;
            // std::cout << "W_CTE: " << _w_cte << std::endl;
            // std::cout << "W_ETHETA: " << _w_etheta << std::endl;
            // std::cout << "W_V: " << _w_vel << std::endl;
            // std::cout << "W_ANGVEL: " << _w_angvel << std::endl;
            // std::cout << "W_DANGVEL: " << _w_angvel_d << std::endl;
            // std::cout << "W_DA: " << _w_linvel_d << std::endl;

            _x_start = 0;
            _y_start = _x_start + _mpc_steps;
            _theta_start = _y_start + _mpc_steps;
            _v_start = _theta_start + _mpc_steps;
            _s_start = _v_start + _mpc_steps;
            _s_dot_start = _s_start + _mpc_steps;
            _angvel_start = _s_dot_start + _mpc_steps;
            _linacc_start = _angvel_start + _mpc_steps - 1;
            _s_ddot_start = _linacc_start + _mpc_steps - 1;

            // cout << "\n!! FG_eval Obj parameters updated !! " << _mpc_steps << endl;
        }

        void set_reference(const std::vector<Spline1D> &reference)
        {
            _reference = reference;
            _use_eigen = true;
        }

        void set_reference(const std::vector<SplineWrapper> &reference)
        {
            _reference_tk = reference;
            _use_eigen = false;
        }

        void set_dist_map(const std::shared_ptr<distmap::DistanceMap> &dist_map)
        {
            std::cout << "[MPCC] Loaded dist map" << std::endl;
            _dist_grid_ptr = dist_map;
        }

        void set_tubes(const std::vector<SplineWrapper> &tubes)
        {
            _tubes = tubes;
        }

        // MPC implementation (cost func & constraints)
        typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
        void operator()(ADvector &fg, const ADvector &vars)
        {
            // cost function is fg[0]
            fg[0] = 0;

            for (int i = 0; i < _mpc_steps; i++)
            {
                AD<double> s = vars[_s_start + i];
                AD<double> sdot = vars[_s_dot_start + i];
                AD<double> x = vars[_x_start + i];
                AD<double> y = vars[_y_start + i];

                AD<double> ref_x;
                AD<double> ref_y;
                AD<double> ref_vx;
                AD<double> ref_vy;
                AD<double> ref_phi;

                if (_use_eigen)
                {
                    ref_x = _reference[0](CppAD::Value(CppAD::Var2Par(s))).coeff(0);
                    ref_y = _reference[1](CppAD::Value(CppAD::Var2Par(s))).coeff(0);
                    ref_vx = _reference[0].derivatives(CppAD::Value(CppAD::Var2Par(s)), 1).coeff(1);
                    ref_vy = _reference[1].derivatives(CppAD::Value(CppAD::Var2Par(s)), 1).coeff(1);
                    ref_phi = CppAD::atan2(ref_vy, ref_vx);
                }
                else
                {
                    ref_x = _reference_tk[0].spline(CppAD::Value(CppAD::Var2Par(s)));
                    ref_y = _reference_tk[1].spline(CppAD::Value(CppAD::Var2Par(s)));
                    ref_vx = _reference_tk[0].spline.deriv(1, CppAD::Value(CppAD::Var2Par(s)));
                    ref_vy = _reference_tk[1].spline.deriv(1, CppAD::Value(CppAD::Var2Par(s)));
                    ref_phi = CppAD::atan2(ref_vy, ref_vx);
                }

                AD<double> e_c = CppAD::sin(ref_phi) * (x - ref_x) - CppAD::cos(ref_phi) * (y - ref_y);
                AD<double> e_l = (-CppAD::cos(ref_phi) * (x - ref_x) - CppAD::sin(ref_phi) * (y - ref_y));

                fg[0] += 5 * CppAD::pow(e_c, 2);
                fg[0] += 50 * CppAD::pow(e_l, 2);
                fg[0] += -sdot;
            }

            // Minimize the use of actuators.
            for (int i = 0; i < _mpc_steps - 1; i++)
            {
                fg[0] += _w_angvel * CppAD::pow(vars[_angvel_start + i], 2);
            }

            // // Minimize the value gap between sequential actuations.
            for (int i = 0; i < _mpc_steps - 2; i++)
            {
                fg[0] += _w_angvel_d * CppAD::pow(vars[_angvel_start + i + 1] - vars[_angvel_start + i], 2);
                fg[0] += _w_linvel_d * CppAD::pow(vars[_linacc_start + i + 1] - vars[_linacc_start + i], 2);
                fg[0] += _w_linvel_d * CppAD::pow(vars[_s_ddot_start + i + 1] - vars[_s_ddot_start + i], 2);
            }

            // fg[x] for constraints
            // Initial constraints
            fg[1 + _x_start] = vars[_x_start];
            fg[1 + _y_start] = vars[_y_start];
            fg[1 + _theta_start] = vars[_theta_start];
            fg[1 + _v_start] = vars[_v_start];
            fg[1 + _s_start] = vars[_s_start];
            fg[1 + _s_dot_start] = vars[_s_dot_start];

            // Add system dynamic model constraint
            for (int i = 0; i < _mpc_steps - 1; i++)
            {
                // The state at time t+1 .
                AD<double> x1 = vars[_x_start + i + 1];
                AD<double> y1 = vars[_y_start + i + 1];
                AD<double> theta1 = vars[_theta_start + i + 1];
                AD<double> v1 = vars[_v_start + i + 1];
                AD<double> s1 = vars[_s_start + i + 1];
                AD<double> sdot1 = vars[_s_dot_start + i + 1];

                // The state at time t.
                AD<double> x0 = vars[_x_start + i];
                AD<double> y0 = vars[_y_start + i];
                AD<double> theta0 = vars[_theta_start + i];
                AD<double> v0 = vars[_v_start + i];
                AD<double> s0 = vars[_s_start + i];
                AD<double> sdot0 = vars[_s_dot_start + i];

                // Only consider the actuation at time t.
                // AD<double> angvel0 = vars[_angvel_start + i];
                AD<double> w0 = vars[_angvel_start + i];
                AD<double> a0 = vars[_linacc_start + i];
                AD<double> sddot0 = vars[_s_ddot_start + i];

                // model equations
                fg[2 + _x_start + i] = x1 - (x0 + v0 * CppAD::cos(theta0) * _dt);
                fg[2 + _y_start + i] = y1 - (y0 + v0 * CppAD::sin(theta0) * _dt);
                fg[2 + _theta_start + i] = theta1 - (theta0 + w0 * _dt);
                fg[2 + _v_start + i] = v1 - (v0 + a0 * _dt);
                fg[2 + _s_start + i] = s1 - (s0 + sdot0 * _dt);
                fg[2 + _s_dot_start + i] = sdot1 - (sdot0 + sddot0 * _dt);

                if (_use_cbf)
                {
                    AD<double> alpha = this->_alpha;
                    AD<double> d2 = this->_colinear;
                    AD<double> padding = this->_padding;

                    AD<double> xdot = v0 * cos(theta0);
                    AD<double> ydot = v0 * sin(theta0);

                    AD<double> dist = _dist_grid_ptr->atPositionSafe(
                                          CppAD::Value(CppAD::Var2Par(x0)), CppAD::Value(CppAD::Var2Par(y0)), true) +
                                      1e-6;
                    AD<double> D = dist - padding - 1e-6;

                    distmap::DistanceMap::Gradient grad = _dist_grid_ptr->gradientAtPosition(
                        CppAD::Value(CppAD::Var2Par(x0)), CppAD::Value(CppAD::Var2Par(y0)), true);
                    AD<double> dx = grad.dx;
                    AD<double> dy = grad.dy;
                    AD<double> grad_norm = CppAD::sqrt(dx * dx + dy * dy);
                    dx *= (-dist / grad_norm);
                    dy *= (-dist / grad_norm);

                    AD<double> p0 = (dx * CppAD::cos(theta0) + dy * CppAD::sin(theta0)) / dist;
                    AD<double> pp = (-dy * CppAD::cos(theta0) + dx * CppAD::sin(theta0)) / dist;
                    AD<double> P = p0 + v0 * d2;

                    AD<double> Lfh1 = (CppAD::exp(-P) * xdot / dist) * (-dx + D * (CppAD::cos(theta0) - dx * p0 / dist));
                    AD<double> Lfh2 = (CppAD::exp(-P) * ydot / dist) * (-dy + D * (CppAD::sin(theta0) - dy * p0 / dist));
                    AD<double> Lfh = Lfh1 + Lfh2;

                    AD<double> Lgh1 = -D * d2 * CppAD::exp(-P);
                    AD<double> Lgh2 = D * pp * CppAD::exp(-P);

                    AD<double> h = D * CppAD::exp(-P);

                    fg[2 + _s_dot_start + _mpc_steps + i] = Lfh + Lgh1 * a0 + Lgh2 * w0 + alpha * h;
                }
            }
        }

    private:
        int _x_start, _y_start, _theta_start, _v_start, _s_start;
        int _s_dot_start, _angvel_start, _linacc_start, _s_ddot_start;

        bool _use_cbf;
        double _alpha;
        double _padding;
        double _colinear;

        std::shared_ptr<distmap::DistanceMap> _dist_grid_ptr;
    };
}

MPCC::MPCC()
{
    // Set default value
    _mpc_steps = 10;
    _max_angvel = 3.0;    // Maximal angvel radian (~30 deg)
    _max_linvel = 2.0;    // Maximal linvel accel
    _max_linacc = 4.0;    // Maximal linacc accel
    _bound_value = 1.0e3; // Bound value for other variables

    _use_cbf = false;
    _alpha = 1.0;
    _colinear = 0.01;
    _padding = .05;

    _x_start = 0;
    _y_start = _x_start + _mpc_steps;
    _theta_start = _y_start + _mpc_steps;
    _v_start = _theta_start + _mpc_steps;
    _s_start = _v_start + _mpc_steps;
    _s_dot_start = _s_start + _mpc_steps;
    _angvel_start = _s_dot_start + _mpc_steps;
    _linacc_start = _angvel_start + _mpc_steps - 1;
    _s_ddot_start = _linacc_start + _mpc_steps - 1;

    mpc_x = {};
    mpc_y = {};
    mpc_theta = {};
    mpc_linvels = {};
    mpc_s = {};
    mpc_s_dot = {};

    mpc_angvels = {};
    mpc_linaccs = {};
    mpc_s_ddots = {};

    _tubes = {};

    _state = Eigen::VectorXd(6);
    _state << 0, 0, 0, 0, 0, 0;

    // _s_dot is the pseudo-state which dictates how quickly to move the reference
    _s_dot = 0;

    _use_eigen = false;
}

MPCC::~MPCC()
{
}

void MPCC::load_params(const std::map<std::string, double> &params)
{
    _params = params;
    // Init parameters for MPC object
    _mpc_steps = _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
    _max_angvel = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL") : _max_angvel;
    _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL") : _max_linvel;
    _max_linacc = _params.find("MAX_LINACC") != _params.end() ? _params.at("MAX_LINACC") : _max_linacc;
    _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND") : _bound_value;

    _use_cbf = params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
    _alpha = params.find("CBF_ALPHA") != params.end() ? params.at("CBF_ALPHA") : _alpha;
    _colinear = params.find("CBF_COLINEAR") != params.end() ? params.at("CBF_COLINEAR") : _colinear;
    _padding = params.find("CBF_PADDING") != params.end() ? params.at("CBF_PADDING") : _padding;

    // std::cerr << "max linvel is " << _max_linvel << std::endl;

    _x_start = 0;
    _y_start = _x_start + _mpc_steps;
    _theta_start = _y_start + _mpc_steps;
    _v_start = _theta_start + _mpc_steps;
    _s_start = _v_start + _mpc_steps;
    _s_dot_start = _s_start + _mpc_steps;
    _angvel_start = _s_dot_start + _mpc_steps;
    _linacc_start = _angvel_start + _mpc_steps - 1;
    _s_ddot_start = _linacc_start + _mpc_steps - 1;

    if (_use_cbf)
        std::cout << "[MPCC] Using CBF!" << std::endl;

    std::cout << "\n!! MPC Obj parameters updated !! " << std::endl;
}

void MPCC::set_tubes(const std::vector<SplineWrapper> &tubes)
{
    _tubes = tubes;
}

void MPCC::set_reference(const std::vector<Spline1D> &reference, double arclen)
{
    _reference = reference;
    _ref_length = arclen;
    _use_eigen = true;
}

void MPCC::set_reference(const std::vector<SplineWrapper> &reference, double arclen)
{
    _reference_tk = reference;
    _ref_length = arclen;
    _use_eigen = false;
}

double MPCC::get_s_from_state(const Eigen::VectorXd &state)
{
    // find the s which minimizes dist to robot
    double s = 0;
    double min_dist = 1e6;
    Eigen::Vector2d pos(state(0), state(1));
    for (double i = 0.0; i < _ref_length; i += .05)
    {
        Eigen::Vector2d p;
        if (_use_eigen)
            p = Eigen::Vector2d(_reference[0](i).coeff(0), _reference[1](i).coeff(0));
        else
            p = Eigen::Vector2d(_reference_tk[0].spline(i), _reference_tk[1].spline(i));

        double d = (pos - p).squaredNorm();
        if (d < min_dist)
        {
            min_dist = d;
            s = i;
        }
    }

    return s;
}

void MPCC::set_dist_map(const std::shared_ptr<distmap::DistanceMap> &dist_map)
{
    _dist_grid_ptr = dist_map;
    std::cout << "[MPC_ACC] Distance map set!" << std::endl;
}

Eigen::VectorXd MPCC::get_state()
{
    return _state;
}

std::vector<double> MPCC::solve(const Eigen::VectorXd &state)
{
    bool ok = true;

    // if (_use_cbf && _dist_grid_ptr.get() == nullptr)
    // {
    //     std::cerr << "Distance map not yet set" << std::endl;
    //     return {0, 0};
    // }

    if (_use_cbf && _tubes.size() == 0)
    {
        std::cerr << "Tubes not set yet" << std::endl;
        return {0, 0};
    }

    typedef CPPAD_TESTVECTOR(double) Dvector;
    const double x = state[0];
    const double y = state[1];
    const double theta = state[2];
    const double v = state[3];

    // pseudo-state variables
    const double s = get_s_from_state(state);
    const double s_dot = _s_dot;

    // Set the number of model variables (includes both states and inputs).
    // For example: If the state is a 4 element vector, the actuators is a 2
    // element vector and there are 10 timesteps. The number of variables is:
    // 4 * 10 + 2 * 9

    size_t n_vars = _mpc_steps * (state.size() + 2) + (_mpc_steps - 1) * 3;

    // Set the number of constraints
    // size_t n_constraints = _mpc_steps * (state.size() + 2);
    // plus 2 because there s and s_dot are not in state variable
    size_t n_constraints = _mpc_steps * (state.size() + 2);

    if (_use_cbf)
        n_constraints += 2*_mpc_steps;

    // std::cerr << "N_VARS: " << n_vars << std::endl;
    // std::cerr << "N_CONSTRAINTS: " << n_constraints << std::endl;

    // Initial value of the independent variables.
    // SHOULD BE 0 besides initial state.
    Dvector vars(n_vars);
    for (int i = 0; i < n_vars; i++)
    {
        vars[i] = 0;
    }

    // Set the initial variable values
    vars[_x_start] = x;
    vars[_y_start] = y;
    vars[_theta_start] = theta;
    vars[_v_start] = v;
    vars[_s_start] = s;
    vars[_s_dot_start] = s_dot;

    // warm start with previous solution
    if (mpc_x.size() > 0)
    {
        for (int i = 1; i < _mpc_steps; ++i)
        {
            vars[_x_start + i] = mpc_x[i];
            vars[_y_start + i] = mpc_y[i];
            vars[_theta_start + i] = mpc_theta[i];
            vars[_v_start + i] = mpc_linvels[i];
            vars[_s_start + i] = mpc_s[i];
            vars[_s_dot_start + i] = mpc_s_dot[i];
        }

        for (int i = 0; i < _mpc_steps - 1; ++i)
        {
            vars[_angvel_start + i] = mpc_angvels[i];
            vars[_linacc_start + i] = mpc_linaccs[i];
            vars[_s_ddot_start + i] = mpc_s_ddots[i];
        }
    }

    // Set lower and upper limits for variables.
    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set all non-actuators upper and lowerlimits
    // to the max negative and positive values.
    for (int i = 0; i < _angvel_start; i++)
    {
        vars_lowerbound[i] = -_bound_value;
        vars_upperbound[i] = _bound_value;
    }

    for (int i = _v_start; i < _s_start; i++)
    {
        // vars_lowerbound[i] = 0; //-_max_linvel;
        vars_lowerbound[i] = -_max_linvel;
        vars_upperbound[i] = _max_linvel;
    }

    for (int i = _s_start; i < _s_dot_start; i++)
    {
        vars_lowerbound[i] = 0;
        vars_upperbound[i] = _ref_length;
    }

    for (int i = _s_dot_start; i < _angvel_start; ++i)
    {
        vars_lowerbound[i] = 0;
        vars_upperbound[i] = _max_linvel;
    }

    // The upper and lower limits of angvel are set to -25 and 25
    // degrees (values in radians).
    for (int i = _angvel_start; i < _linacc_start; i++)
    {
        vars_lowerbound[i] = -_max_angvel;
        vars_upperbound[i] = _max_angvel;
    }

    // Acceleration/decceleration upper and lower limits
    // these limits also apply to _s_ddot
    for (int i = _linacc_start; i < n_vars; i++)
    {
        vars_lowerbound[i] = -_max_linacc;
        vars_upperbound[i] = _max_linacc;
    }

    // Lower and upper limits for the constraints
    // Should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);

    for (int i = 0; i < n_constraints; i++)
    {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }

    if (_use_cbf)
    {
        for (int i = n_constraints - 2*_mpc_steps; i < n_constraints; i++)
        {
            constraints_lowerbound[i] = 0.0;
            constraints_upperbound[i] = 1.0e19;
        }
    }

    constraints_lowerbound[_x_start] = x;
    constraints_lowerbound[_y_start] = y;
    constraints_lowerbound[_theta_start] = theta;
    constraints_lowerbound[_v_start] = v;
    constraints_lowerbound[_s_start] = s;
    constraints_lowerbound[_s_dot_start] = s_dot;

    constraints_upperbound[_x_start] = x;
    constraints_upperbound[_y_start] = y;
    constraints_upperbound[_theta_start] = theta;
    constraints_upperbound[_v_start] = v;
    constraints_upperbound[_s_start] = s;
    constraints_upperbound[_s_dot_start] = s_dot;

    // object that computes objective and constraints
    FG_eval fg_eval;
    fg_eval.load_params(_params);
    if (_use_eigen)
        fg_eval.set_reference(_reference);
    else
        fg_eval.set_reference(_reference_tk);

    // if (_use_cbf)
    //     fg_eval.set_dist_map(_dist_grid_ptr);
    if (_use_cbf)
        fg_eval.set_tubes(_tubes);

    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          .5\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    std::cout << "starting solver" << std::endl;
    CppAD::ipopt::solve<Dvector, FG_eval>(
        options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
        constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= (solution.status == CppAD::ipopt::solve_result<Dvector>::success || solution.status == CppAD::ipopt::solve_result<Dvector>::stop_at_acceptable_point || solution.status == CppAD::ipopt::solve_result<Dvector>::feasible_point_found);

    if (!ok)
    {
        std::cerr << "Error occured during solve (" << solution.status << ")" << std::endl;
        // print out state
        solution.x[_angvel_start] = 0;
        solution.x[_linacc_start] = 0;
    }

    // Cost
    auto cost = solution.obj_value;
    // std::cout << "------------ Total Cost(solution): " << cost << " ------------" << std::endl;
    // std::cout << "max_angvel:" << _max_angvel <<std::endl;
    // std::cout << "max_linvel:" << _max_linvel <<std::endl;

    // std::cout << "-----------------------------------------------" <<std::endl;

    mpc_x = {};
    mpc_y = {};
    mpc_theta = {};
    mpc_linvels = {};
    mpc_s = {};
    mpc_s_dot = {};

    for (int i = 0; i < _mpc_steps; i++)
    {
        mpc_x.push_back(solution.x[_x_start + i]);
        mpc_y.push_back(solution.x[_y_start + i]);
        mpc_theta.push_back(solution.x[_theta_start + i]);
        mpc_linvels.push_back(solution.x[_v_start + i]);
        mpc_s.push_back(solution.x[_s_start + i]);
        mpc_s_dot.push_back(solution.x[_s_dot_start + i]);
    }

    mpc_angvels = {};
    mpc_linaccs = {};
    mpc_s_ddots = {};
    for (int i = 0; i < _mpc_steps - 1; i++)
    {
        mpc_angvels.push_back(solution.x[_angvel_start + i]);
        mpc_linaccs.push_back(solution.x[_linacc_start + i]);
        mpc_s_ddots.push_back(solution.x[_s_ddot_start + i]);
    }

    std::cout << "MPC INPUTS ARE: " << std::endl;
    std::cout << "acc: " << solution.x[_linacc_start] << std::endl;
    std::cout << "anvel: " << solution.x[_angvel_start] << std::endl;
    std::cout << "s ddot: " << solution.x[_s_ddot_start] << std::endl;

    _s_dot = solution.x[_s_dot_start];

    std::vector<double> result;
    result.push_back(solution.x[_angvel_start]);
    result.push_back(solution.x[_linacc_start]);

    _state << solution.x[_x_start],
              solution.x[_y_start],
              solution.x[_theta_start],
              solution.x[_v_start],
              solution.x[_s_start],
              solution.x[_s_dot_start];

    return result;
}
