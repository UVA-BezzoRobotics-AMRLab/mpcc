#include <iostream>
#include <uav_mpc/mpcc_acados.h>

std::vector<Spline1D> create_reference()
{
    std::vector<Spline1D> ret;

    Eigen::RowVectorXd ss(11);
    ss << 0.0, 1.88503, 3.77006, 5.655, 7.54, 9.42515, 11.31, 13.1952, 15.08, 16.9652, 18.85;

    Eigen::RowVectorXd xs(11);
    xs << 0.0003, 1.7632, 2.8531, 2.8531, 1.7632, -0.0003, -1.7632, -2.8531, -2.8531, -1.7632, -0.0003;

    Eigen::RowVectorXd ys(11);
    ys << 0.0, -0.5728, -2.0729, -3.9271, -5.4271, -6.0, -5.4271, -3.9271, -2.0729, -0.5728, 0.0;


    const auto fitX = SplineFitting1D::Interpolate(xs, 3, ss);
    Spline1D splineX = Spline1D(fitX);

    const auto fitY = SplineFitting1D::Interpolate(ys, 3, ss);
    Spline1D splineY = Spline1D(fitY);

    std::cout << splineX.knots() << std::endl;
    std::cout << splineX.ctrls() << std::endl;

    ret.push_back(splineX);
    ret.push_back(splineY);

    return ret;
}

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
}

MPCC::~MPCC()
{
    if (_acados_ocp_capsule)
        delete _acados_ocp_capsule;

    if (_new_time_steps)
        delete _new_time_steps;
}

void MPCC::set_tubes(const std::vector<SplineWrapper> &tubes)
{
    return;
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

    if(_new_time_steps)
        delete [] _new_time_steps;

    _new_time_steps = new double[_mpc_steps];
    for(int i = 0; i < _mpc_steps; ++i)
    {
        _new_time_steps[i] = _dt * i;
    }

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

    std::cout << "getting s" << std::endl;
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

    std::cout << "done" << std::endl;
    return s;
}

std::vector<double> MPCC::solve(const Eigen::VectorXd &state)
{
    // set initial condition
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
    lbx0[4] = get_s_from_state(state);
    ubx0[4] = get_s_from_state(state);
    lbx0[5] = _s_dot;
    ubx0[5] = _s_dot;

    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx", ubx0);

    // warm start
    double x_init[NX];
    x_init[0] = lbx0[0];
    x_init[1] = lbx0[1];
    x_init[2] = lbx0[2];
    x_init[3] = lbx0[3];
    x_init[4] = lbx0[4];
    x_init[5] = lbx0[5];

    double u_init[NU];
    u_init[0] = 0.0;
    u_init[1] = 0.0;
    u_init[2] = 0.0;

    for(int i = 0; i < _mpc_steps; ++i)
    {
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
    }

    // no input in final step of MPC
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);
    std::cout << "solving" << std::endl;
    int status = unicycle_model_mpcc_acados_solve(_acados_ocp_capsule);
    std::cout << "done solving" << std::endl;

    if (status == ACADOS_SUCCESS)
        std::cout << "unicycle_model_mpcc_acados_solve(): SUCCESS!" << std::endl;
    else
        std::cout << "unicycle_model_mpcc_acados_solve() failed with status " << status << std::endl;

    return {};
}

int main()
{
    MPCC obj;
    std::map<std::string, double> params;

    double arclen = 18.85;
    std::vector<Spline1D> ref = create_reference();
    obj.set_reference(ref, arclen);

    obj.load_params(params);
    Eigen::VectorXd state(4);
    state << 0,0,0,0;


    obj.solve(state);

    

    return 0;
}
