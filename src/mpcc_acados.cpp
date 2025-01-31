#include <iostream>

#include <uav_mpc/utils.h>
#include <uav_mpc/mpcc_acados.h>

MPCC::MPCC()
{
    // Set default value
    _dt = .05;
    _mpc_steps = 20;
    _max_angvel = 3.0;    // Maximal angvel radian (~30 deg)
    _max_linvel = 2.0;    // Maximal linvel accel
    _max_linacc = 4.0;    // Maximal linacc accel
    _bound_value = 1.0e3; // Bound value for other variables

    _w_ql = 50.0;
    _w_qc = .1;
    _w_q_speed = .3;

    _use_cbf = false;
    _alpha = 1.0;
    _colinear = 0.01;
    _padding = .05;

    _ref_samples = 10;
    _ref_len_sz = 4.0;

    _use_eigen = false;

    _acados_ocp_capsule = nullptr;
    _new_time_steps = nullptr;

    _s_dot = 0;
    _ref_length = 0;

    _state = Eigen::VectorXd(NX);
    for (int i = 0; i < NX; ++i)
        _state(i) = 0;

    _prev_x0.resize((_mpc_steps + 1) * NX);
    _prev_u0.resize(_mpc_steps * NU);

    // _x_start = 0;
    // _y_start = _x_start + _mpc_steps;
    // _theta_start = _y_start + _mpc_steps;
    // _v_start = _theta_start + _mpc_steps;
    // _s_start = _v_start + _mpc_steps;
    // _s_dot_start = _s_start + _mpc_steps;
    // _angvel_start = _s_dot_start + _mpc_steps;
    // _linacc_start = _angvel_start + _mpc_steps - 1;
    // _s_ddot_start = _linacc_start + _mpc_steps - 1;

    _x_start = 0;
    _y_start = 1;
    _theta_start = 2;
    _v_start = 3;
    _s_start = 4;
    _s_dot_start = 5;
    _angvel_start = 6;
    _linacc_start = 7;
    _s_ddot_start = 8;
    _ind_state_inc = 6;
    _ind_input_inc = 3;

    _is_shift_warm = false;

    // cpg_update_A_mat(0, -1.1);
}

MPCC::~MPCC()
{
    if (_acados_ocp_capsule)
        delete _acados_ocp_capsule;

    if (_new_time_steps)
        delete _new_time_steps;
}

// void MPCC::set_tubes(const std::vector<Spline1D> &tubes)
void MPCC::set_tubes(const std::vector<Eigen::VectorXd> &tubes)
{
    _tubes = tubes;
}

void MPCC::load_params(const std::map<std::string, double> &params)
{
    _params = params;

    // Init parameters for MPC object
    _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
    _mpc_steps = _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
    _max_angvel = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL") : _max_angvel;
    _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL") : _max_linvel;
    _max_linacc = _params.find("MAX_LINACC") != _params.end() ? _params.at("MAX_LINACC") : _max_linacc;
    _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND") : _bound_value;

    _w_angvel = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL") : _w_angvel;
    _w_angvel_d = params.find("W_DANGVEL") != params.end() ? params.at("W_DANGVEL") : _w_angvel_d;
    _w_linvel_d = params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

    _w_ql = params.find("W_LAG") != params.end() ? params.at("W_LAG") : _w_ql;
    _w_qc = params.find("W_CONTOUR") != params.end() ? params.at("W_CONTOUR") : _w_qc;
    _w_q_speed = params.find("W_SPEED") != params.end() ? params.at("W_SPEED") : _w_q_speed;

    _ref_len_sz = params.find("REF_LENGTH") != params.end() ? params.at("REF_LENGTH") : _ref_len_sz;
    _ref_samples = params.find("REF_SAMPLES") != params.end() ? params.at("REF_SAMPLES") : _ref_samples;

    _use_cbf = params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
    _alpha = params.find("CBF_ALPHA") != params.end() ? params.at("CBF_ALPHA") : _alpha;
    _colinear = params.find("CBF_COLINEAR") != params.end() ? params.at("CBF_COLINEAR") : _colinear;
    _padding = params.find("CBF_PADDING") != params.end() ? params.at("CBF_PADDING") : _padding;

    _acados_ocp_capsule = unicycle_model_mpcc_acados_create_capsule();

    if (_new_time_steps)
        delete[] _new_time_steps;

    // _new_time_steps = new double[_mpc_steps];
    // for(int i = 0; i < _mpc_steps; ++i)
    // {
    //     _new_time_steps[i] = _dt * i;
    //     std::cout << _new_time_steps[i] << std::endl;
    // }

    _acados_sim_capsule = unicycle_model_mpcc_acados_sim_solver_create_capsule();
    int status = unicycle_model_mpcc_acados_sim_create(_acados_sim_capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    // acados sim
    _sim_in = unicycle_model_mpcc_acados_get_sim_in(_acados_sim_capsule);
    _sim_out = unicycle_model_mpcc_acados_get_sim_out(_acados_sim_capsule);
    _sim_dims = unicycle_model_mpcc_acados_get_sim_dims(_acados_sim_capsule);
    _sim_config = unicycle_model_mpcc_acados_get_sim_config(_acados_sim_capsule);

    status = unicycle_model_mpcc_acados_create_with_discretization(_acados_ocp_capsule, _mpc_steps, _new_time_steps);
    if (status)
    {
        std::cout << "unicycle_model_mpcc_acados_create() returned status " << status << ". Exiting." << std::endl;
        exit(1);
    }

    // acados solver
    _nlp_in = unicycle_model_mpcc_acados_get_nlp_in(_acados_ocp_capsule);
    _nlp_out = unicycle_model_mpcc_acados_get_nlp_out(_acados_ocp_capsule);
    _nlp_opts = unicycle_model_mpcc_acados_get_nlp_opts(_acados_ocp_capsule);
    _nlp_dims = unicycle_model_mpcc_acados_get_nlp_dims(_acados_ocp_capsule);
    _nlp_solver = unicycle_model_mpcc_acados_get_nlp_solver(_acados_ocp_capsule);
    _nlp_config = unicycle_model_mpcc_acados_get_nlp_config(_acados_ocp_capsule);

    // _x_start = 0;
    // _y_start = _x_start + _mpc_steps;
    // _theta_start = _y_start + _mpc_steps;
    // _v_start = _theta_start + _mpc_steps;
    // _s_start = _v_start + _mpc_steps;
    // _s_dot_start = _s_start + _mpc_steps;
    // _angvel_start = _s_dot_start + _mpc_steps;
    // _linacc_start = _angvel_start + _mpc_steps - 1;
    // _s_ddot_start = _linacc_start + _mpc_steps - 1;

    _prev_x0.resize((_mpc_steps + 1) * NX);
    _prev_u0.resize(_mpc_steps * NU);

    _x_start = 0;
    _y_start = 1;
    _theta_start = 2;
    _v_start = 3;
    _s_start = 4;
    _s_dot_start = 5;
    _angvel_start = 6;
    _linacc_start = 7;
    _s_ddot_start = 8;
    _ind_state_inc = 6;
    _ind_input_inc = 3;

    std::cout << "!! MPC Obj parameters updated !! " << std::endl;
    std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

void MPCC::set_reference(const std::vector<Spline1D> &reference, double arclen)
{
    _reference = reference;
    _ref_length = arclen;
    return;
}

double MPCC::get_s_from_state(const Eigen::VectorXd &state)
{

    // find the s which minimizes dist to robot
    double s = 0;
    double min_dist = 1e6;
    Eigen::Vector2d pos(state(0), state(1));
    for (double i = 0.0; i < _ref_length; i += .05)
    {
        Eigen::Vector2d p = Eigen::Vector2d(_reference[0](i).coeff(0), _reference[1](i).coeff(0));

        double d = (pos - p).squaredNorm();
        if (d < min_dist)
        {
            min_dist = d;
            s = i;
        }
    }

    return s;
}

Eigen::VectorXd MPCC::get_state()
{
    return _state;
}

Eigen::VectorXd MPCC::next_state(const Eigen::VectorXd &current_state, const Eigen::VectorXd &control)
{
    Eigen::VectorXd ret(NX);

    // Extracting current state values
    double x1 = current_state(0);
    double y1 = current_state(1);
    double theta1 = current_state(2);
    double v1 = current_state(3);
    double s1 = current_state(4);
    double sdot1 = current_state(5);

    // Extracting control inputs
    double a = control(0);
    double w = control(1);
    double sddot = control(2);

    // Dynamics equations
    ret(0) = x1 + v1 * cos(theta1) * _dt;
    ret(1) = y1 + v1 * sin(theta1) * _dt;
    ret(2) = theta1 + w * _dt;
    ret(3) = v1 + a * _dt;
    ret(4) = s1 + sdot1 * _dt;
    ret(5) = sdot1 + sddot * _dt;

    return ret;
}

std::vector<Spline1D> MPCC::get_ref_from_s(double s)
{
    // get reference for next _ref_len_sz meters, indexing from s=0 onwards
    // need to also down sample the tubes
    Eigen::RowVectorXd ss, xs, ys; //, abvs, blws;
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
            // abvs(i) = _tubes[0](ss(i) + s).coeff(0);
            // blws(i) = _tubes[1](ss(i) + s).coeff(0);
        }
        else
        {
            // xs(i) = dx * (ss(i) + s - _ref_length) + px;
            // ys(i) = dy * (ss(i) + s - _ref_length) + py;
            xs(i) = px;
            ys(i) = py;
            // abvs(i) = final_abv_d;
            // blws(i) = final_blw_d;
        }
    }

    // fit splines
    const auto fitX = utils::Interp(xs, 3, ss);
    Spline1D splineX(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    Spline1D splineY(fitY);

    // const auto fitAbv = utils::Interp(abvs, 3, ss);
    // Spline1D splineAbv(fitAbv);

    // const auto fitBlw = utils::Interp(blws, 3, ss);
    // Spline1D splineBlw(fitBlw);

    // return {splineX, splineY, splineAbv, splineBlw};
    return {splineX, splineY};
}

void MPCC::warm_start_no_u(double *x_init)
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

// From linear to nonlinear MPC: bridging the gap via the real-time iteration, Gros et. al.
void MPCC::warm_start_shifted_u()
{
    double starting_s = _prev_x0[1 * NX + 4];
    for (int i = 1; i < _mpc_steps; ++i)
    {
        Eigen::VectorXd warm_state = _prev_x0.segment(i * NX, NX);
        warm_state(4) -= starting_s;

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "x", &warm_state[0]);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "u", &_prev_u0[i * NU]);

        // double xr = ref[0](warm_state(4)).coeff(0);
        // double yr = ref[1](warm_state(4)).coeff(0);
        // double tx = ref[0].derivatives(warm_state(4), 1).coeff(1);
        // double ty = ref[1].derivatives(warm_state(4), 1).coeff(1);

        // double signed_d = (-(warm_state[0] - xr) * ty + (warm_state[1] - yr) * tx) / sqrt(tx*tx + ty*ty);
        // double cons_upper = signed_d - utils::eval_traj(_tubes[0], warm_state(4));
        // double cons_lower = utils::eval_traj(_tubes[1], warm_state(4)) - signed_d;

        // if (cons_upper > -1e-1 || cons_lower > -1e-1)
        // {
        //     std::cout << "upper constraint value is: " << cons_upper << std::endl;
        //     std::cout << "lower constraint value is: " << cons_lower << std::endl;
        // }
    }

    Eigen::VectorXd xN_prev = _prev_x0.tail(NX);
    xN_prev(4) -= starting_s;

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x", &xN_prev[0]);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u", &_prev_u0[(_mpc_steps - 1) * NU]);

    Eigen::VectorXd uN_prev = _prev_u0.tail(NU);
    Eigen::VectorXd xN = next_state(xN_prev, uN_prev);

    // double xr = ref[0](xN(4)).coeff(0);
    // double yr = ref[1](xN(4)).coeff(0);
    // double tx = ref[0].derivatives(xN(4), 1).coeff(1);
    // double ty = ref[1].derivatives(xN(4), 1).coeff(1);

    // double signed_d = (-(xN[0] - xr) * ty + (xN[1] - yr) * tx) / sqrt(tx*tx + ty*ty);
    // double cons_upper = signed_d - utils::eval_traj(_tubes[0], xN(4));
    // double cons_lower = utils::eval_traj(_tubes[1], xN(4)) - signed_d;

    // if (cons_upper > -1e-1 || cons_lower > -1e-1)
    // {
    //     std::cout << "upper constraint value is: " << cons_upper << std::endl;
    //     std::cout << "lower constraint value is: " << cons_lower << std::endl;
    // }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &xN[0]);
}

bool MPCC::set_solver_parameters(const std::vector<Spline1D> &ref)
{
    double params[NP];
    auto ctrls_x = ref[0].ctrls();
    auto ctrls_y = ref[1].ctrls();

    int num_params = ctrls_x.size() + ctrls_y.size() + _tubes[0].size() + _tubes[1].size() + 3;
    if (num_params != NP)
    {
        std::cout << "reference size " << num_params << " does not match acados parameter size " << NP << std::endl;
        return false;
    }

    params[NP - 3] = _w_qc;
    params[NP - 2] = _w_ql;
    params[NP - 1] = _w_q_speed;

    // std::cout << "w_qc: " << _w_qc << std::endl;
    // std::cout << "w_ql: " << _w_ql << std::endl;
    // std::cout << "w_q_speed: " << _w_q_speed << std::endl;

    for (int i = 0; i < ctrls_x.size(); ++i)
    {
        params[i] = ctrls_x[i];
        params[i + ctrls_x.size()] = ctrls_y[i];
    }

    for (int i = 0; i < _tubes[0].size(); ++i)
    {
        params[i + 2 * ctrls_x.size()] = _tubes[0](i);
        params[i + 2 * ctrls_x.size() + _tubes[0].size()] = _tubes[1](i);
    }

    for (int i = 0; i < _mpc_steps + 1; ++i)
        unicycle_model_mpcc_acados_update_params(_acados_ocp_capsule, i, params, NP);

    return true;
}


void MPCC::process_solver_output()
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

    // for (int i = 0; i < xtraj.size(); ++i)
    //     _prev_x0[i] = xtraj[i];

    _prev_x0 = xtraj;
    _prev_u0 = utraj;

    mpc_x = {};
    mpc_y = {};
    mpc_theta = {};
    mpc_linvels = {};
    mpc_s = {};
    mpc_s_dot = {};
    for (int i = 0; i < _mpc_steps; i++)
    {
        mpc_x.push_back(xtraj[_x_start + i * _ind_state_inc]);
        mpc_y.push_back(xtraj[_y_start + i * _ind_state_inc]);
        mpc_theta.push_back(xtraj[_theta_start + i * _ind_state_inc]);
        mpc_linvels.push_back(xtraj[_v_start + i * _ind_state_inc]);
        mpc_s.push_back(xtraj[_s_start + i * _ind_state_inc]);
        mpc_s_dot.push_back(xtraj[_s_dot_start + i * _ind_state_inc]);
    }

    mpc_angvels = {};
    mpc_linaccs = {};
    mpc_s_ddots = {};
    for (int i = 0; i < _mpc_steps - 1; i++)
    {
        mpc_angvels.push_back(xtraj[_angvel_start + i * _ind_input_inc]);
        mpc_linaccs.push_back(xtraj[_linacc_start + i * _ind_input_inc]);
        mpc_s_ddots.push_back(xtraj[_s_ddot_start + i * _ind_input_inc]);
    }


}

std::vector<double> MPCC::solve(const Eigen::VectorXd &state)
{

    if (_tubes.size() == 0)
    {
        std::cout << "tubes are not set yet, mpc cannot run" << std::endl;
        return {0, 0};
    }

    /*************************************
    ********** INITIAL CONDITION *********
    **************************************/

    double lbx0[NBX0];
    double ubx0[NBX0];

    Eigen::VectorXd x0(NBX0);
    x0 << state(0), state(1), state(2), state(3), 0, _s_dot;

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
    double s = get_s_from_state(state);
    std::vector<Spline1D> ref = get_ref_from_s(s);

    double starting_s = _prev_x0[1 * NX + 4];
    if (mpc_x.size() == 0)
        warm_start_no_u(x_init);
    else
        warm_start_shifted_u();

    /*************************************
    ********* SET REFERENCE PARAMS *******
    **************************************/

    if (!set_solver_parameters(ref))
        return {0, 0};

    /*************************************
    ************* RUN SOLVER *************
    **************************************/

    double elapsed_time = 0.0;
    double timer;

    // run at most 2 times, if first fails, try with simple initialization
    for(int i = 0; i < 2; ++i)
    {
        int status = unicycle_model_mpcc_acados_solve(_acados_ocp_capsule);
        ocp_nlp_get(_nlp_config, _nlp_solver, "time_tot", &timer);
        elapsed_time += timer;

        if (status == ACADOS_SUCCESS)
        {
            std::cout << "unicycle_model_mpcc_acados_solve(): SUCCESS! " << elapsed_time * 1000 << std::endl;
            break;
        }
        else if (!_is_shift_warm)
        {
            std::cout << "unicycle_model_mpcc_acados_solve() failed with status " << status << std::endl;
            warm_start_no_u(x_init);
        }
    }

    // unicycle_model_mpcc_acados_print_stats(_acados_ocp_capsule);

    /*************************************
    *********** PROCESS OUTPUT ***********
    **************************************/

    process_solver_output();

    _s_dot = _prev_x0[_s_dot_start + 1 * _ind_state_inc];

    _state << _prev_x0[_x_start],
              _prev_x0[_y_start],
              _prev_x0[_theta_start],
              _prev_x0[_v_start],
              s,
              _prev_x0[_s_dot_start];

    _is_shift_warm = true;

    return {_prev_u0[1], _prev_u0[0]};
}
