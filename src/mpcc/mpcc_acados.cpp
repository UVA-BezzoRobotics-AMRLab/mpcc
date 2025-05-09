#include <mpcc/mpcc_acados.h>
#include <mpcc/utils.h>

#include <array>
#include <chrono>
#include <iostream>
#include <mpcc/termcolor.hpp>

#include "Eigen/src/Core/Matrix.h"

MPCC::MPCC()
{
    // Set default value
    _dt          = .05;
    _mpc_steps   = 20;
    _max_angvel  = 3.0;    // Maximal angvel radian (~30 deg)
    _max_linvel  = 2.0;    // Maximal linvel accel
    _max_linacc  = 4.0;    // Maximal linacc accel
    _bound_value = 1.0e3;  // Bound value for other variables

    _w_ql      = 50.0;
    _w_qc      = .1;
    _w_q_speed = .3;
    _w_ql_lyap = 1;
    _w_qc_lyap = 1;

    _gamma     = .5;
    _use_cbf   = false;
    _alpha_abv = 1.0;
    _alpha_blw = 1.0;
    _colinear  = 0.01;
    _padding   = .05;

    _use_eigen   = false;
    _ref_samples = 10;
    _ref_len_sz  = 4.0;
    _ref_length  = 0;

    _acados_ocp_capsule = nullptr;
    _new_time_steps     = nullptr;

    _s_dot = 0;

    _state = Eigen::VectorXd(NX);
    for (int i = 0; i < NX; ++i) _state(i) = 0;

    _prev_x0.resize((_mpc_steps + 1) * NX);
    _prev_u0.resize(_mpc_steps * NU);

    mpc_x.resize(_mpc_steps);
    mpc_y.resize(_mpc_steps);
    mpc_theta.resize(_mpc_steps);
    mpc_linvels.resize(_mpc_steps);
    mpc_s.resize(_mpc_steps);
    mpc_s_dot.resize(_mpc_steps);

    mpc_angvels.resize(_mpc_steps - 1);
    mpc_linaccs.resize(_mpc_steps - 1);
    mpc_s_ddots.resize(_mpc_steps - 1);

    _x_start       = 0;
    _y_start       = 1;
    _theta_start   = 2;
    _v_start       = 3;
    _s_start       = 4;
    _s_dot_start   = 5;
    _angvel_start  = 6;
    _linacc_start  = 7;
    _s_ddot_start  = 8;
    _ind_state_inc = 6;
    _ind_input_inc = 3;

    _use_dyna_obs  = false;
    _is_shift_warm = false;
    _solve_success = false;

    _odom = Eigen::VectorXd(3);

    // cpg_update_A_mat(0, -1.1);
}

MPCC::~MPCC()
{
    if (_acados_ocp_capsule) delete _acados_ocp_capsule;

    if (_new_time_steps) delete _new_time_steps;
}

// void MPCC::set_tubes(const std::vector<Spline1D> &tubes)
void MPCC::set_tubes(const std::array<Eigen::VectorXd, 2>& tubes) { _tubes = tubes; }

void MPCC::load_params(const std::map<std::string, double>& params)
{
    _params = params;

    // Init parameters for MPC object
    _dt         = params.find("DT") != params.end() ? params.at("DT") : _dt;
    _mpc_steps  = _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
    _max_angvel = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL") : _max_angvel;
    _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL") : _max_linvel;
    _max_linacc =
        _params.find("MAX_LINACC") != _params.end() ? _params.at("MAX_LINACC") : _max_linacc;
    _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND") : _bound_value;

    _w_angvel = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL") : _w_angvel;
    _w_angvel_d =
        params.find("W_DANGVEL") != params.end() ? params.at("W_DANGVEL") : _w_angvel_d;
    _w_linvel_d = params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

    _w_ql      = params.find("W_LAG") != params.end() ? params.at("W_LAG") : _w_ql;
    _w_qc      = params.find("W_CONTOUR") != params.end() ? params.at("W_CONTOUR") : _w_qc;
    _w_q_speed = params.find("W_SPEED") != params.end() ? params.at("W_SPEED") : _w_q_speed;

    _ref_len_sz =
        params.find("REF_LENGTH") != params.end() ? params.at("REF_LENGTH") : _ref_len_sz;
    _ref_samples =
        params.find("REF_SAMPLES") != params.end() ? params.at("REF_SAMPLES") : _ref_samples;

    _gamma     = params.find("CLF_GAMMA") != params.end() ? params.at("CLF_GAMMA") : _gamma;
    _w_ql_lyap = params.find("CLF_W_LAG") != params.end() ? params.at("CLF_W_LAG") : _w_ql_lyap;
    _w_qc_lyap =
        params.find("CLF_W_CONTOUR") != params.end() ? params.at("CLF_W_CONTOUR") : _w_ql_lyap;

    _use_cbf = params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
    _alpha_abv =
        params.find("CBF_ALPHA_ABV") != params.end() ? params.at("CBF_ALPHA_ABV") : _alpha_abv;
    _alpha_blw =
        params.find("CBF_ALPHA_BLW") != params.end() ? params.at("CBF_ALPHA_BLW") : _alpha_blw;
    _colinear =
        params.find("CBF_COLINEAR") != params.end() ? params.at("CBF_COLINEAR") : _colinear;
    _padding = params.find("CBF_PADDING") != params.end() ? params.at("CBF_PADDING") : _padding;

    _acados_ocp_capsule = unicycle_model_mpcc_acados_create_capsule();

    if (_new_time_steps) delete[] _new_time_steps;

    _acados_sim_capsule = unicycle_model_mpcc_acados_sim_solver_create_capsule();
    int status          = unicycle_model_mpcc_acados_sim_create(_acados_sim_capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    // acados sim
    _sim_in     = unicycle_model_mpcc_acados_get_sim_in(_acados_sim_capsule);
    _sim_out    = unicycle_model_mpcc_acados_get_sim_out(_acados_sim_capsule);
    _sim_dims   = unicycle_model_mpcc_acados_get_sim_dims(_acados_sim_capsule);
    _sim_config = unicycle_model_mpcc_acados_get_sim_config(_acados_sim_capsule);

    status = unicycle_model_mpcc_acados_create_with_discretization(_acados_ocp_capsule,
                                                                   _mpc_steps, _new_time_steps);
    if (status)
    {
        std::cout << "unicycle_model_mpcc_acados_create() returned status " << status
                  << ". Exiting." << std::endl;
        exit(1);
    }

    // acados solver
    _nlp_in     = unicycle_model_mpcc_acados_get_nlp_in(_acados_ocp_capsule);
    _nlp_out    = unicycle_model_mpcc_acados_get_nlp_out(_acados_ocp_capsule);
    _nlp_opts   = unicycle_model_mpcc_acados_get_nlp_opts(_acados_ocp_capsule);
    _nlp_dims   = unicycle_model_mpcc_acados_get_nlp_dims(_acados_ocp_capsule);
    _nlp_solver = unicycle_model_mpcc_acados_get_nlp_solver(_acados_ocp_capsule);
    _nlp_config = unicycle_model_mpcc_acados_get_nlp_config(_acados_ocp_capsule);

    mpc_x.resize(_mpc_steps);
    mpc_y.resize(_mpc_steps);
    mpc_theta.resize(_mpc_steps);
    mpc_linvels.resize(_mpc_steps);
    mpc_s.resize(_mpc_steps);
    mpc_s_dot.resize(_mpc_steps);

    mpc_angvels.resize(_mpc_steps - 1);
    mpc_linaccs.resize(_mpc_steps - 1);
    mpc_s_ddots.resize(_mpc_steps - 1);

    _prev_x0.resize((_mpc_steps + 1) * NX);
    _prev_u0.resize(_mpc_steps * NU);

    _x_start       = 0;
    _y_start       = 1;
    _theta_start   = 2;
    _v_start       = 3;
    _s_start       = 4;
    _s_dot_start   = 5;
    _angvel_start  = 6;
    _linacc_start  = 7;
    _s_ddot_start  = 8;
    _ind_state_inc = 6;
    _ind_input_inc = 3;

    std::cout << "!! MPC Obj parameters updated !! " << std::endl;
    std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

void MPCC::set_reference(const std::array<Spline1D, 2>& reference, double arclen)
{
    _reference  = reference;
    _ref_length = arclen;
    return;
}

double MPCC::get_s_from_state(const Eigen::VectorXd& state)
{
    // find the s which minimizes dist to robot
    double s        = 0;
    double min_dist = 1e6;
    Eigen::Vector2d pos(state(0), state(1));
    for (double i = 0.0; i < _ref_length; i += .05)
    {
        Eigen::Vector2d p =
            Eigen::Vector2d(_reference[0](i).coeff(0), _reference[1](i).coeff(0));

        double d = (pos - p).squaredNorm();
        if (d < min_dist)
        {
            min_dist = d;
            s        = i;
        }
    }

    return s;
}

void MPCC::set_dyna_obs(const Eigen::MatrixXd& dyna_obs)
{
    _use_dyna_obs = true;
    _dyna_obs     = dyna_obs;
}

void MPCC::set_odom(const Eigen::VectorXd& odom) { _odom = odom; }

const Eigen::VectorXd& MPCC::get_state() const { return _state; }

const bool MPCC::get_solver_status() const { return _solve_success; }

Eigen::VectorXd MPCC::next_state(const Eigen::VectorXd& current_state,
                                 const Eigen::VectorXd& control)
{
    Eigen::VectorXd ret(NX);

    // Extracting current state values
    double x1     = current_state(0);
    double y1     = current_state(1);
    double theta1 = current_state(2);
    double v1     = current_state(3);
    double s1     = current_state(4);
    double sdot1  = current_state(5);

    // Extracting control inputs
    double a     = control(0);
    double w     = control(1);
    double sddot = control(2);

    // Dynamics equations
    ret(0) = x1 + v1 * cos(theta1) * _dt;
    ret(1) = y1 + v1 * sin(theta1) * _dt;
    ret(2) = theta1 + w * _dt;
    ret(3) = std::max(std::min(v1 + a * _dt, _max_linvel), -_max_linvel);
    ret(4) = s1 + sdot1 * _dt;
    ret(5) = std::max(std::min(sdot1 + sddot * _dt, _max_linvel), -_max_linvel);

    return ret;
}

std::array<Spline1D, 2> MPCC::compute_adjusted_ref(double s) const
{
    // get reference for next _ref_len_sz meters, indexing from s=0 onwards
    // need to also down sample the tubes
    Eigen::RowVectorXd ss, xs, ys;  //, abvs, blws;
    ss.resize(_ref_samples);
    xs.resize(_ref_samples);
    ys.resize(_ref_samples);

    double px = _reference[0](_ref_length).coeff(0);
    double py = _reference[1](_ref_length).coeff(0);

    // capture reference at each sample
    for (int i = 0; i < _ref_samples; ++i)
    {
        // ss(i) = ((double)i) * 4. / 10.;
        ss(i) = ((double)i) * _ref_len_sz / (_ref_samples - 1);

        // if sample domain exceeds trajectory, duplicate final point
        if (ss(i) + s <= _ref_length)
        {
            xs(i) = _reference[0](ss(i) + s).coeff(0);
            ys(i) = _reference[1](ss(i) + s).coeff(0);
        }
        else
        {
            xs(i) = px;
            ys(i) = py;
        }
    }

    // fit splines
    const auto fitX = utils::Interp(xs, 3, ss);
    Spline1D splineX(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    Spline1D splineY(fitY);

    return {splineX, splineY};
}

void MPCC::warm_start_no_u(double* x_init)
{
    double u_init[NU];
    u_init[0] = 0.0;
    u_init[1] = 0.0;
    u_init[2] = 0.0;

    for (int i = 0; i < _mpc_steps; ++i)
    {
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
    }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);
}

// From linear to nonlinear MPC: bridging the gap via the real-time iteration,
// Gros et. al.
void MPCC::warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd& state)
{
    double starting_s = _prev_x0[1 * NX + 4];
    if (correct_perturb)
    {
        std::cout << termcolor::red << "[MPCC] Guess pos. too far, correcting"
                  << termcolor::reset << std::endl;

        Eigen::VectorXd curr = state;

        // project forward previous control inputs, starting from true current
        // state
        for (int i = 0; i < _mpc_steps - 1; ++i)
        {
            ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", &curr[0]);
            ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", &_prev_u0[(i + 1) * NU]);
            curr = next_state(curr, _prev_u0.segment((i + 1) * NU, NU));
            // std::cout << curr.transpose() << std::endl;
        }

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x", &curr[0]);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                        &_prev_u0[(_mpc_steps - 1) * NU]);

        curr = next_state(curr, _prev_u0.tail(NU));
        // std::cout << curr.transpose() << std::endl;

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &curr[0]);
        // exit(0);
    }
    else
    {
        for (int i = 1; i < _mpc_steps; ++i)
        {
            Eigen::VectorXd warm_state = _prev_x0.segment(i * NX, NX);
            warm_state(4) -= starting_s;

            ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "x", &warm_state[0]);
            ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "u", &_prev_u0[i * NU]);
        }

        Eigen::VectorXd xN_prev = _prev_x0.tail(NX);
        xN_prev(4) -= starting_s;

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x", &xN_prev[0]);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                        &_prev_u0[(_mpc_steps - 1) * NU]);

        Eigen::VectorXd uN_prev = _prev_u0.tail(NU);
        Eigen::VectorXd xN      = next_state(xN_prev, uN_prev);

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &xN[0]);
    }
}

bool MPCC::set_solver_parameters(const std::array<Spline1D, 2>& adjusted_ref)
{
    double params[NP];
    auto ctrls_x = adjusted_ref[0].ctrls();
    auto ctrls_y = adjusted_ref[1].ctrls();

    int num_params = ctrls_x.size() + ctrls_y.size() + _tubes[0].size() + _tubes[1].size() + 10;
    if (num_params != NP)
    {
        std::cout << "[MPCC] reference size " << num_params
                  << " does not match acados parameter size " << NP << std::endl;
        return false;
    }

    params[NP - 10] = _w_qc;
    params[NP - 9]  = _w_ql;
    params[NP - 8]  = _w_q_speed;
    params[NP - 7]  = _alpha_abv;
    params[NP - 6]  = _alpha_blw;
    params[NP - 5]  = _w_qc_lyap;
    params[NP - 4]  = _w_ql_lyap;
    params[NP - 3]  = _gamma;
    params[NP - 2]  = 1e3;
    params[NP - 1]  = 1e3;

    for (int i = 0; i < ctrls_x.size(); ++i)
    {
        params[i]                  = ctrls_x[i];
        params[i + ctrls_x.size()] = ctrls_y[i];
    }

    for (int i = 0; i < _tubes[0].size(); ++i)
    {
        params[i + 2 * ctrls_x.size()]                    = _tubes[0](i);
        params[i + 2 * ctrls_x.size() + _tubes[0].size()] = _tubes[1](i);
    }

    Eigen::VectorXd obs_pos, obs_vel;
    if (_use_dyna_obs)
    {
        obs_pos = _dyna_obs.col(0);
        obs_vel = _dyna_obs.col(1);
    }

    for (int i = 0; i < _mpc_steps + 1; ++i)
    {
        if (_use_dyna_obs)
        {
            params[NP - 2] = obs_pos(0);
            params[NP - 1] = obs_pos(1);
            std::cout << termcolor::red << "[MPCC] obs_pos: " << obs_pos.transpose()
                      << termcolor::reset << std::endl;
        }
        unicycle_model_mpcc_acados_update_params(_acados_ocp_capsule, i, params, NP);

        obs_pos += obs_vel * _dt;
    }

    return true;
}

void MPCC::process_solver_output(double s)
{
    // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
    Eigen::VectorXd xtraj((_mpc_steps + 1) * NX);
    Eigen::VectorXd utraj(_mpc_steps * NU);
    for (int i = 0; i < _mpc_steps; ++i)
    {
        ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "x", &xtraj[i * NX]);
        ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "u", &utraj[i * NU]);
    }

    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &xtraj[_mpc_steps * NX]);

#ifdef NS
    Eigen::VectorXd slacks(NS);
    // std::cout << "getting slacks " << NS << std::endl;
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, 1, "sl", &slacks[0]);

    std::cout << "[MPCC] Slack values are: " << slacks.transpose() << std::endl;
#endif

    _prev_x0 = xtraj;
    _prev_u0 = utraj;

    for (int i = 0; i < _mpc_steps; ++i)
    {
        mpc_x[i]       = xtraj[_x_start + i * _ind_state_inc];
        mpc_y[i]       = xtraj[_y_start + i * _ind_state_inc];
        mpc_theta[i]   = xtraj[_theta_start + i * _ind_state_inc];
        mpc_linvels[i] = xtraj[_v_start + i * _ind_state_inc];
        mpc_s[i]       = xtraj[_s_start + i * _ind_state_inc] + s;
        mpc_s_dot[i]   = xtraj[_s_dot_start + i * _ind_state_inc];
    }

    for (int i = 0; i < _mpc_steps - 1; ++i)
    {
        mpc_angvels[i] = xtraj[_angvel_start + i * _ind_input_inc];
        mpc_linaccs[i] = xtraj[_linacc_start + i * _ind_input_inc];
        mpc_s_ddots[i] = xtraj[_s_ddot_start + i * _ind_input_inc];
    }
}

void MPCC::reset_horizon()
{
    for (int i = 0; i < _mpc_steps; ++i)
    {
        mpc_x[i]       = _odom(0);
        mpc_y[i]       = _odom(1);
        mpc_theta[i]   = 0;
        mpc_linvels[i] = 0;
        mpc_s[i]       = 0;
        mpc_s_dot[i]   = 0;
    }

    for (int i = 0; i < _mpc_steps - 1; ++i)
    {
        mpc_angvels[i] = 0;
        mpc_linaccs[i] = 0;
        mpc_s_ddots[i] = 0;
    }
}

std::array<double, 2> MPCC::solve(const Eigen::VectorXd& state, bool is_reverse)
{
    _solve_success = false;

    if (_tubes.size() == 0)
    {
        std::cout << "[MPCC] tubes are not set yet, mpc cannot run" << std::endl;
        return {0, 0};
    }

    /*************************************
    ********** INITIAL CONDITION *********
    **************************************/

    // _s_dot = std::min(std::max((s - _state(5)) / _dt, 0.), _max_linvel);
    if (state.size() != NBX0)
    {
        std::cout << termcolor::yellow << "[MPCC] state sized passed has size " << state.size()
                  << " but should be " << NBX0 << termcolor::reset << std::endl;
        return {0, 0};
    }

    double lbx0[NBX0];
    double ubx0[NBX0];

    // Eigen::VectorXd x0(NBX0);
    // x0 << state(0), state(1), state(2), state(3), 0, _s_dot;
    Eigen::VectorXd x0 = state;
    if (x0(2) < 0) x0(2) += 2 * M_PI;

    memcpy(lbx0, &x0[0], NBX0 * sizeof(double));
    memcpy(ubx0, &x0[0], NBX0 * sizeof(double));

    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx", ubx0);

    /*************************************
    ********* INITIALIZE SOLUTION ********
    **************************************/

    // in our case NX = NBX0
    double x_init[NX];
    memcpy(x_init, lbx0, NX * sizeof(double));

    double u_init[NU];
    u_init[0] = 0.0;
    u_init[1] = 0.0;
    u_init[2] = 0.0;

    // generate params from reference trajectory starting at current s
    double s                             = get_s_from_state(state);
    std::array<Spline1D, 2> adjusted_ref = compute_adjusted_ref(s);

    // Eigen::Vector2d prev_pos = _prev_x0.head(2);
    Eigen::Vector2d prev_pos = _prev_x0.segment(NX, 2);
    Eigen::Vector2d curr_pos = state.head(2);

    double dist = (prev_pos - curr_pos).norm();
    if (_is_shift_warm && dist > 1e-1)
    {
        std::cout << termcolor::red << "[MPCC] Pos too far (" << dist
                  << "), turning off shifted warm start" << std::endl;
        std::cout << "[MPCC] x0: " << x0.transpose() << termcolor::reset << std::endl;
        // _is_shift_warm = false;
    }

    double starting_s = _prev_x0[1 * NX + 4];
    if (!_is_shift_warm)
        warm_start_no_u(x_init);
    else
    {
        // warm_start_shifted_u(false, x0);
        warm_start_shifted_u((prev_pos - curr_pos).norm() > 5e-2, x0);
    }

    /*************************************
    ********* SET REFERENCE PARAMS *******
    **************************************/

    if (!set_solver_parameters(adjusted_ref)) return {0, 0};

    /*************************************
    ************* RUN SOLVER *************
    **************************************/

    // double elapsed_time = 0.0;
    double timer;

    // run at most 2 times, if first fails, try with simple initialization
    for (int i = 0; i < 2; ++i)
    {
        // timer for acados using chrono
        auto start = std::chrono::high_resolution_clock::now();
        int status = unicycle_model_mpcc_acados_solve(_acados_ocp_capsule);
        auto end   = std::chrono::high_resolution_clock::now();
        // for some reason this causes problems in docker container, commenting
        // out for now
        // ocp_nlp_get(_nlp_config, _nlp_solver, "time_tot", &timer);
        // elapsed_time += timer;

        if (status == ACADOS_SUCCESS)
        {
            std::cout << "[MPCC] unicycle_model_mpcc_acados_solve(): SUCCESS! "
                      << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
            //          << elapsed_time * 1000 << std::endl;
            _is_shift_warm = true;
            _solve_success = true;
            break;
        }
        else
        {
            _is_shift_warm = false;
            std::cout << "[MPCC] unicycle_model_mpcc_acados_solve() failed with "
                         "status "
                      << status << std::endl;
            std::cout << "[MPCC] using simple warm start procedure" << std::endl;
            std::cout << "[MPCC] xinit is: " << x0.transpose() << std::endl;
            warm_start_no_u(x_init);
        }
    }

    /*************************************
    *********** PROCESS OUTPUT ***********
    **************************************/

    process_solver_output(s);
    std::cout << "mpc x[0] is " << _prev_x0.head(NX).transpose() << std::endl;
    std::cout << "true x[0] is " << x0.transpose() << std::endl;
    std::cout << "mpc x[1] is " << _prev_x0.segment(NX, NX).transpose() << std::endl;

    // unicycle_model_mpcc_acados_print_stats(_acados_ocp_capsule);
    for (int i = 0; i < mpc_x.size(); ++i)
    {
        double si     = mpc_s[i];
        double x      = mpc_x[i];
        double y      = mpc_y[i];
        double xr     = adjusted_ref[0](si).coeff(0);
        double yr     = adjusted_ref[1](si).coeff(0);
        double xr_dot = adjusted_ref[0].derivatives(si, 1).coeff(1);
        double yr_dot = adjusted_ref[1].derivatives(si, 1).coeff(1);

        double den      = xr_dot * xr_dot + yr_dot * yr_dot;
        double obs_dirx = -yr_dot / den;
        double obs_diry = xr_dot / den;

        double signed_d = (x - xr) * obs_dirx + (y - yr) * obs_diry;
    }

    _state << _prev_x0[_x_start], _prev_x0[_y_start], _prev_x0[_theta_start],
        _prev_x0[_v_start], s, _prev_x0[_s_dot_start];

    /*if (is_reverse) _prev_u0[0] *= -1;*/

    return {_prev_u0[1], _prev_u0[0]};
}

Eigen::VectorXd MPCC::get_cbf_data(const Eigen::VectorXd& state, const Eigen::VectorXd& control,
                                   bool is_abv) const
{
    Eigen::VectorXd ret_data(3);
    double s = 0;  // state(4);

    Eigen::VectorXd coeffs;
    if (is_abv)
        coeffs = _tubes[0];
    else
        coeffs = _tubes[1];

    double tube_dist = 0;
    double x_pow     = 1;

    for (int i = 0; i < coeffs.size(); ++i)
    {
        tube_dist += coeffs[i] * x_pow;
        x_pow *= s;
    }

    std::array<Spline1D, 2> adjusted_ref = compute_adjusted_ref(s);
    double xr                            = adjusted_ref[0](s).coeff(0);
    double yr                            = adjusted_ref[1](s).coeff(0);

    double xr_dot = adjusted_ref[0].derivatives(s, 1).coeff(1);
    double yr_dot = adjusted_ref[1].derivatives(s, 1).coeff(1);

    double den      = sqrt(xr_dot * xr_dot + yr_dot * yr_dot);
    double obs_dirx = -yr_dot / den;
    double obs_diry = xr_dot / den;

    double signed_d = (state(0) - xr) * obs_dirx + (state(1) - yr) * obs_diry;
    double p        = obs_dirx * cos(state(2)) + obs_diry * sin(state(2)) + state(3) * .05;

    double h_val;
    if (is_abv)
        h_val = (tube_dist - signed_d - .1) * exp(-p);
    else
        h_val = (signed_d - tube_dist - .1) * exp(-p);

    signed_d = is_abv ? signed_d : -signed_d;
    if (h_val > 100)
    {
        std::cout << termcolor::yellow << "ref length is " << _ref_length << std::endl;
        std::cout << "s: " << s << " h_val: " << h_val << " is abv: " << is_abv
                  << termcolor::reset << std::endl;
        std::cout << "tube dist: " << tube_dist << " signed_d: " << signed_d << std::endl;
        exit(-1);
    }

    return Eigen::Vector3d(h_val, signed_d, atan2(obs_diry, obs_dirx));
}

// utility
void MPCC::apply_affine_transform(Eigen::VectorXd& state, const Eigen::Vector2d& rot_point,
                                  const Eigen::MatrixXd& m_affine)
{
    Eigen::Vector3d state_ext(0, 0, 1);
    state_ext.head(2) = state.head(2);

    state_ext.head(2) -= rot_point;
    state_ext = m_affine * state_ext;
    state_ext.head(2) += rot_point;

    state.head(2) = state_ext.head(2);
}
