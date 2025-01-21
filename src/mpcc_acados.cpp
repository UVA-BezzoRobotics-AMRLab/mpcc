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

    _use_cbf = false;
    _alpha = 1.0;
    _colinear = 0.01;
    _padding = .05;

    _use_eigen = false;

    _acados_ocp_capsule = nullptr;
    _new_time_steps = nullptr;

    _s_dot = 0;
    _ref_length = 0;

    _state = Eigen::VectorXd(6);
    _state << 0, 0, 0, 0, 0, 0;

    _prev_x0.resize((_mpc_steps + 1) * _state.size());

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
}

MPCC::~MPCC()
{
    if (_acados_ocp_capsule)
        delete _acados_ocp_capsule;

    if (_new_time_steps)
        delete _new_time_steps;
}

void MPCC::set_tubes(const std::vector<Spline1D> &tubes)
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

    int status = unicycle_model_mpcc_acados_create_with_discretization(_acados_ocp_capsule, _mpc_steps, _new_time_steps);
    if (status)
    {
        std::cout << "unicycle_model_mpcc_acados_create() returned status " << status << ". Exiting." << std::endl;
        exit(1);
    }

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

    _prev_x0.resize((_mpc_steps + 1) * _state.size());

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

std::vector<Spline1D> MPCC::get_ref_from_s(double s)
{
    // get reference for next 4 meters, indexing from s=0 onwards
    // need to also down sample the tubes
    Eigen::RowVectorXd ss, xs, ys, abvs, blws;
    ss.resize(11);
    xs.resize(11);
    ys.resize(11);
    abvs.resize(11);
    blws.resize(11);

    double px = _reference[0](_ref_length).coeff(0);
    double py = _reference[1](_ref_length).coeff(0);

    double final_abv_d = _tubes[0](4).coeff(0);
    double final_blw_d = _tubes[1](4).coeff(0);

    // double dx = _reference[0].derivatives(_ref_length, 1).coeff(1);
    // double dy = _reference[1].derivatives(_ref_length, 1).coeff(1);

    // capture reference at each sample
    for (int i = 0; i < 11; ++i)
    {
        ss(i) = ((double)i) * 4. / 10.;

        // if sample domain exceeds trajectory, duplicate final point
        if (ss(i) + s <= _ref_length)
        {
            xs(i) = _reference[0](ss(i) + s).coeff(0);
            ys(i) = _reference[1](ss(i) + s).coeff(0);
            abvs(i) = _tubes[0](ss(i) + s).coeff(0);
            blws(i) = _tubes[1](ss(i) + s).coeff(0);
        }
        else
        {
            // xs(i) = dx * (ss(i) + s - _ref_length) + px;
            // ys(i) = dy * (ss(i) + s - _ref_length) + py;
            xs(i) = px;
            ys(i) = py;
            abvs(i) = final_abv_d;
            blws(i) = final_blw_d;
        }
    }

    // fit splines
    const auto fitX = utils::Interp(xs, 3, ss);
    Spline1D splineX(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    Spline1D splineY(fitY);

    const auto fitAbv = utils::Interp(abvs, 3, ss);
    Spline1D splineAbv(fitAbv);

    const auto fitBlw = utils::Interp(blws, 3, ss);
    Spline1D splineBlw(fitBlw);

    return {splineX, splineY, splineAbv, splineBlw};
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
    lbx0[0] = state[0];
    ubx0[0] = state[0];
    lbx0[1] = state[1];
    ubx0[1] = state[1];
    lbx0[2] = state[2];
    ubx0[2] = state[2];
    lbx0[3] = state[3];
    ubx0[3] = state[3];
    lbx0[4] = 0;
    ubx0[4] = 0;
    lbx0[5] = _s_dot;
    ubx0[5] = _s_dot;

    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx", ubx0);

    /*************************************
    ********* INITIALIZE SOLUTION ********
    **************************************/

    double x_init[NX];
    x_init[0] = lbx0[0];
    x_init[1] = lbx0[1];
    x_init[2] = lbx0[2];
    x_init[3] = lbx0[3];
    x_init[4] = 0;
    x_init[5] = lbx0[5];

    // for (int i = 0; i < NX; ++i)
    // {
    //     std::cout << "x_i[" << i << "]: " << x_init[i] << "\tubx[" << i << "]: " << ubx0[i] << std::endl;
    // }

    double u_init[NU];
    u_init[0] = 0.0;
    u_init[1] = 0.0;
    u_init[2] = 0.0;

    for (int i = 0; i < _mpc_steps; ++i)
    {
        // double x_stage[NX];
        // for(int j = 0; j < NX; ++j)
        //     x_stage[j] = _prev_x0[(i+1)*_ind_state_inc + j];

        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
    }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);

    /*************************************
    ********* SET REFERENCE PARAMS *******
    **************************************/

    // generate params from reference trajectory starting at current s
    double s = get_s_from_state(state);
    std::vector<Spline1D> ref = get_ref_from_s(s);

    double params[NP];
    auto ctrls_x = ref[0].ctrls();
    auto ctrls_y = ref[1].ctrls();
    auto ctrls_abv = ref[2].ctrls();
    auto ctrls_blw = ref[3].ctrls();

    if (ctrls_x.size() + ctrls_y.size() + ctrls_abv.size() + ctrls_blw.size() != NP)
    {
        std::cout << "reference size does not match acados parameter size" << std::endl;
        return {0, 0};
    }

    for (int i = 0; i < ctrls_x.size(); ++i)
    {
        params[i] = ctrls_x[i];
        params[i + ctrls_x.size()] = ctrls_y[i];
        params[i + 2 * ctrls_x.size()] = ctrls_abv[i];
        params[i + 3 * ctrls_x.size()] = ctrls_blw[i];
    }

    for (int i = 0; i < _mpc_steps + 1; ++i)
        unicycle_model_mpcc_acados_update_params(_acados_ocp_capsule, i, params, NP);

    /*************************************
    ************* RUN SOLVER *************
    **************************************/

    double elapsed_time;

    int status = unicycle_model_mpcc_acados_solve(_acados_ocp_capsule);
    ocp_nlp_get(_nlp_config, _nlp_solver, "time_tot", &elapsed_time);

    if (status == ACADOS_SUCCESS)
        std::cout << "unicycle_model_mpcc_acados_solve(): SUCCESS! " << elapsed_time * 1000 << std::endl;
    else
    {
        std::cout << "unicycle_model_mpcc_acados_solve() failed with status " << status << std::endl;
        return {0, 0};
    }

    // unicycle_model_mpcc_acados_print_stats(_acados_ocp_capsule);

    /*************************************
    *********** PROCESS OUTPUT ***********
    **************************************/

    Eigen::VectorXd utraj(NU);
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, 0, "u", &utraj[0]);

    // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
    Eigen::VectorXd xtraj((_mpc_steps + 1) * NX);
    for (int i = 0; i <= _mpc_steps; ++i)
        ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "x", &xtraj[i * NX]);

    for (int i = 0; i < xtraj.size(); ++i)
        _prev_x0[i] = xtraj[i];

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

    _s_dot = xtraj[_s_dot_start + 1 * _ind_state_inc];

    // double xr = ref[0](0).coeff(0);
    // double yr = ref[1](0).coeff(0);
    // double tx = ref[0].derivatives(0, 1).coeff(1);
    // double ty = ref[1].derivatives(0, 1).coeff(1);
    // double phi = atan2(ty, tx);

    // double el = -cos(phi) * (state[0] - xr) - sin(phi) * (state[1] - yr);
    // std::cout << "lag error is: " << el << std::endl;

    // double *sl;
    // double *inequality_residuals;  // Replace m with the number of inequality constraints
    // ocp_nlp_get(_nlp_config, _nlp_solver, "sl", sl);
    // ocp_nlp_get(_nlp_config, _nlp_solver, "ineq_residuals", &inequality_residuals);

    // std::cout << "ineq constraints" << std::endl;
    // for (int i = 0; i < 2; i++)
    // {
    //     std::cout << "ineq_residual[" << i <<"] = " << inequality_residuals[i] << std::endl;
    //     // std::cout << i << " slack value " << sl[i] << std::endl;
    // }


    // since s always = 0 in mpc state, need to use manually computed s
    _state << xtraj[_x_start],
        xtraj[_y_start],
        xtraj[_theta_start],
        xtraj[_v_start],
        s,
        xtraj[_s_dot_start];

    return {utraj[1], utraj[0]};
}

// int main()
// {
//     MPCC obj;
//     std::map<std::string, double> params;

//     double arclen = 18.85;
//     std::vector<Spline1D> ref = create_reference();
//     obj.set_reference(ref, arclen);

//     obj.load_params(params);
//     Eigen::VectorXd state(4);
//     state << 0, 0, 0, 0;

//     obj.solve(state);

//     return 0;
// }
