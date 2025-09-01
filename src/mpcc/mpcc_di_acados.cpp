#include <mpcc/mpcc_di_acados.h>
#include <mpcc/utils.h>
#include <cmath>
#include <mpcc/termcolor.hpp>

DIMPCC::DIMPCC() {
  _dt         = .05;
  _max_linvel = 2.;
  _max_linacc = 2.;
  _mpc_steps  = 20;

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

  _s_dot       = 0;
  _ref_samples = 10;
  _ref_len_sz  = 4.0;
  _ref_length  = 0;

  _has_run       = false;
  _solve_success = false;
  _is_shift_warm = false;

  _acados_ocp_capsule = nullptr;
  _new_time_steps     = nullptr;

  _state = Eigen::VectorXd::Zero(kNX);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  mpc_x.resize(_mpc_steps);
  mpc_y.resize(_mpc_steps);
  mpc_vx.resize(_mpc_steps);
  mpc_vy.resize(_mpc_steps);
  mpc_s.resize(_mpc_steps);
  mpc_s_dot.resize(_mpc_steps);

  mpc_ax.resize(_mpc_steps - 1);
  mpc_ay.resize(_mpc_steps - 1);
  mpc_s_ddots.resize(_mpc_steps - 1);
}

DIMPCC::~DIMPCC() {
  if (_acados_ocp_capsule)
    delete _acados_ocp_capsule;
  if (_new_time_steps)
    delete _new_time_steps;
}

void DIMPCC::load_params(const std::map<std::string, double>& params) {
  _params = params;

  // Init parameters for MPC object
  _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
  _mpc_steps =
      _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
  _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL")
                                                        : _max_linvel;
  _max_linacc = _params.find("MAX_LINACC") != _params.end()
                    ? _params.at("MAX_LINACC")
                    : _max_linacc;

  _w_ql = params.find("W_LAG") != params.end() ? params.at("W_LAG") : _w_ql;
  _w_qc =
      params.find("W_CONTOUR") != params.end() ? params.at("W_CONTOUR") : _w_qc;
  _w_q_speed = params.find("W_SPEED") != params.end() ? params.at("W_SPEED")
                                                      : _w_q_speed;

  _ref_len_sz  = params.find("REF_LENGTH") != params.end()
                     ? params.at("REF_LENGTH")
                     : _ref_len_sz;
  _ref_samples = params.find("REF_SAMPLES") != params.end()
                     ? params.at("REF_SAMPLES")
                     : _ref_samples;

  _gamma     = params.find("CLF_GAMMA") != params.end() ? params.at("CLF_GAMMA")
                                                        : _gamma;
  _w_ql_lyap = params.find("CLF_W_LAG") != params.end() ? params.at("CLF_W_LAG")
                                                        : _w_ql_lyap;
  _w_qc_lyap = params.find("CLF_W_CONTOUR") != params.end()
                   ? params.at("CLF_W_CONTOUR")
                   : _w_ql_lyap;

  _use_cbf =
      params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
  _alpha_abv = params.find("CBF_ALPHA_ABV") != params.end()
                   ? params.at("CBF_ALPHA_ABV")
                   : _alpha_abv;
  _alpha_blw = params.find("CBF_ALPHA_BLW") != params.end()
                   ? params.at("CBF_ALPHA_BLW")
                   : _alpha_blw;
  _colinear  = params.find("CBF_COLINEAR") != params.end()
                   ? params.at("CBF_COLINEAR")
                   : _colinear;
  _padding   = params.find("CBF_PADDING") != params.end()
                   ? params.at("CBF_PADDING")
                   : _padding;

  _acados_ocp_capsule = double_integrator_mpcc_acados_create_capsule();

  if (_new_time_steps)
    delete[] _new_time_steps;

  _acados_sim_capsule =
      double_integrator_mpcc_acados_sim_solver_create_capsule();
  int status = double_integrator_mpcc_acados_sim_create(_acados_sim_capsule);

  if (status) {
    throw std::runtime_error("acados_create() returned status " +
                             std::to_string(status) + ". Exiting.");
  }

  // acados sim
  _sim_in   = double_integrator_mpcc_acados_get_sim_in(_acados_sim_capsule);
  _sim_out  = double_integrator_mpcc_acados_get_sim_out(_acados_sim_capsule);
  _sim_dims = double_integrator_mpcc_acados_get_sim_dims(_acados_sim_capsule);
  _sim_config =
      double_integrator_mpcc_acados_get_sim_config(_acados_sim_capsule);

  status = double_integrator_mpcc_acados_create_with_discretization(
      _acados_ocp_capsule, _mpc_steps, _new_time_steps);

  if (status) {
    throw std::runtime_error(
        "double_integrator_mpcc_acados_create() returned status " +
        std::to_string(status) + ". Exiting.");
  }

  // acados solver
  _nlp_in   = double_integrator_mpcc_acados_get_nlp_in(_acados_ocp_capsule);
  _nlp_out  = double_integrator_mpcc_acados_get_nlp_out(_acados_ocp_capsule);
  _nlp_opts = double_integrator_mpcc_acados_get_nlp_opts(_acados_ocp_capsule);
  _nlp_dims = double_integrator_mpcc_acados_get_nlp_dims(_acados_ocp_capsule);
  _nlp_solver =
      double_integrator_mpcc_acados_get_nlp_solver(_acados_ocp_capsule);
  _nlp_config =
      double_integrator_mpcc_acados_get_nlp_config(_acados_ocp_capsule);

  mpc_x.resize(_mpc_steps);
  mpc_y.resize(_mpc_steps);
  mpc_vx.resize(_mpc_steps);
  mpc_vy.resize(_mpc_steps);
  mpc_s.resize(_mpc_steps);
  mpc_s_dot.resize(_mpc_steps);

  mpc_ax.resize(_mpc_steps - 1);
  mpc_ay.resize(_mpc_steps - 1);
  mpc_s_ddots.resize(_mpc_steps - 1);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

void DIMPCC::reset_horizon() {
  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_x[i]     = _odom(0);
    mpc_y[i]     = _odom(1);
    mpc_vx[i]    = 0;
    mpc_vy[i]    = 0;
    mpc_s[i]     = 0;
    mpc_s_dot[i] = 0;
  }

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    mpc_ax[i]      = 0;
    mpc_ay[i]      = 0;
    mpc_s_ddots[i] = 0;
  }
}

void DIMPCC::set_odom(const Eigen::VectorXd& odom) {}

Eigen::VectorXd DIMPCC::get_cbf_data(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control,
                                     bool is_abv) const {
  Eigen::VectorXd ret_data(3);
  double s = 0;  // state(4);

  Eigen::VectorXd coeffs;
  if (is_abv)
    coeffs = _tubes[0];
  else
    coeffs = _tubes[1];

  double tube_dist = 0;
  double x_pow     = 1;

  for (int i = 0; i < coeffs.size(); ++i) {
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

  double vx = state(2);
  double vy = state(3);
  if (fabs(vx) < 1e-3 && fabs(vy) < 1e-3)
    vx = 1e-3;

  double vel = sqrt(vx * vx + vy * vy);

  double signed_d = (state(0) - xr) * obs_dirx + (state(1) - yr) * obs_diry;
  double p        = (obs_dirx * vx + obs_diry * vy) / vel + vel * .05;

  double h_val;
  if (is_abv)
    h_val = (tube_dist - signed_d - .1) * exp(-p);
  else
    h_val = (signed_d - tube_dist - .1) * exp(-p);

  signed_d = is_abv ? signed_d : -signed_d;
  if (h_val > 100) {
    std::cout << termcolor::yellow << "ref length is " << _ref_length
              << std::endl;
    std::cout << "s: " << s << " h_val: " << h_val << " is abv: " << is_abv
              << termcolor::reset << std::endl;
    std::cout << "tube dist: " << tube_dist << " signed_d: " << signed_d
              << std::endl;
    exit(-1);
  }

  return Eigen::Vector3d(h_val, signed_d, atan2(obs_diry, obs_dirx));
}

double DIMPCC::get_s_from_state(const Eigen::VectorXd& state) {
  // find the s which minimizes dist to robot
  double s        = 0;
  double min_dist = 1e6;
  Eigen::Vector2d pos(state(0), state(1));
  for (double i = 0.0; i < _ref_length; i += .05) {
    Eigen::Vector2d p =
        Eigen::Vector2d(_reference[0](i).coeff(0), _reference[1](i).coeff(0));

    double d = (pos - p).squaredNorm();
    if (d < min_dist) {
      min_dist = d;
      s        = i;
    }
  }

  return s;
}

Eigen::VectorXd DIMPCC::next_state(const Eigen::VectorXd& current_state,
                                   const Eigen::VectorXd& control) {
  Eigen::VectorXd ret(kNX);

  double x1    = current_state(0);
  double y1    = current_state(1);
  double vx1   = current_state(2);
  double vy1   = current_state(3);
  double s1    = current_state(4);
  double sdot1 = current_state(5);

  double ax    = control(0);
  double ay    = control(1);
  double sddot = control(2);

  ret(0) = x1 + vx1 * _dt;
  ret(1) = y1 + vy1 * _dt;
  ret(2) = std::max(std::min(vx1 + ax * _dt, _max_linvel), -_max_linvel);
  ret(3) = std::max(std::min(vy1 + ay * _dt, _max_linvel), -_max_linvel);
  ret(4) = s1 + sdot1 * _dt;
  ret(5) = std::max(std::min(sdot1 + sddot * _dt, _max_linvel), -_max_linvel);

  return ret;
}

void DIMPCC::warm_start_no_u(double* x_init) {
  double u_init[kNU];
  u_init[0] = 0.0;
  u_init[1] = 0.0;
  u_init[2] = 0.0;

  x_init[5] = x_init[3];

  for (int i = 0; i < _mpc_steps; ++i) {
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
  }

  ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);
}

void DIMPCC::warm_start_shifted_u(bool correct_perturb,
                                  const Eigen::VectorXd& state) {
  double starting_s = _prev_x0[1 * kNX + 4];
  if (correct_perturb) {
    std::cout << termcolor::red << "[MPCC] Guess pos. too far, correcting"
              << termcolor::reset << std::endl;

    Eigen::VectorXd curr = state;

    // project forward previous control inputs, starting from true current
    // state
    for (int i = 0; i < _mpc_steps - 1; ++i) {
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", &curr[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u",
                      &_prev_u0[(i + 1) * kNU]);
      curr = next_state(curr, _prev_u0.segment((i + 1) * kNU, kNU));
      // std::cout << curr.transpose() << std::endl;
    }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                    &curr[0]);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                    &_prev_u0[(_mpc_steps - 1) * kNU]);

    curr = next_state(curr, _prev_u0.tail(kNU));
    // std::cout << curr.transpose() << std::endl;

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                    &curr[0]);
    // exit(0);
  } else {
    for (int i = 1; i < _mpc_steps; ++i) {
      Eigen::VectorXd warm_state = _prev_x0.segment(i * kNX, kNX);
      warm_state(4) -= starting_s;

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "x",
                      &warm_state[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "u",
                      &_prev_u0[i * kNU]);
    }

    Eigen::VectorXd xN_prev = _prev_x0.tail(kNX);
    xN_prev(4) -= starting_s;

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                    &xN_prev[0]);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                    &_prev_u0[(_mpc_steps - 1) * kNU]);

    Eigen::VectorXd uN_prev = _prev_u0.tail(kNU);
    Eigen::VectorXd xN      = next_state(xN_prev, uN_prev);

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &xN[0]);
  }
}

bool DIMPCC::set_solver_parameters(
    const std::array<Spline1D, 2>& adjusted_ref) {
  double params[kNP];
  auto ctrls_x = adjusted_ref[0].ctrls();
  auto ctrls_y = adjusted_ref[1].ctrls();

  int num_params =
      ctrls_x.size() + ctrls_y.size() + _tubes[0].size() + _tubes[1].size() + 8;
  if (num_params != kNP) {
    std::cout << "[MPCC] reference size " << num_params
              << " does not match acados parameter size " << kNP << std::endl;
    return false;
  }

  params[kNP - 8] = _w_qc;
  params[kNP - 7] = _w_ql;
  params[kNP - 6] = _w_q_speed;
  params[kNP - 5] = _alpha_abv;
  params[kNP - 4] = _alpha_blw;
  params[kNP - 3] = _w_qc_lyap;
  params[kNP - 2] = _w_ql_lyap;
  params[kNP - 1] = _gamma;

  for (int i = 0; i < ctrls_x.size(); ++i) {
    params[i]                  = ctrls_x[i];
    params[i + ctrls_x.size()] = ctrls_y[i];
  }

  for (int i = 0; i < _tubes[0].size(); ++i) {
    params[i + 2 * ctrls_x.size()]                    = _tubes[0](i);
    params[i + 2 * ctrls_x.size() + _tubes[0].size()] = _tubes[1](i);
  }

  for (int i = 0; i < _mpc_steps + 1; ++i)
    double_integrator_mpcc_acados_update_params(_acados_ocp_capsule, i, params,
                                                kNP);

  return true;
}

std::array<double, 2> DIMPCC::solve(const Eigen::VectorXd& state,
                                    bool is_reverse) {
  _solve_success = false;

  if (_tubes.size() == 0) {
    std::cout << "[MPCC] tubes are not set yet, mpc cannot run" << std::endl;
    return {0, 0};
  }

  /*************************************
  ********** INITIAL CONDITION *********
  **************************************/
  std::cout << termcolor::yellow << "RUNNING IN DIMPCC SOLVE!\n"
            << termcolor::reset;

  if (state.size() != kNBX0) {
    std::cout << termcolor::yellow << "[MPCC] state sized passed has size "
              << state.size() << " but should be " << kNBX0 << termcolor::reset
              << std::endl;
    return {0, 0};
  }

  double lbx0[kNBX0];
  double ubx0[kNBX0];

  // Eigen::VectorXd x0(kNBX0);
  // x0 << state(0), state(1), state(2), state(3), 0, _s_dot;
  Eigen::VectorXd x0 = state;
  if (x0.segment(2, 2).norm() < 1e-3) {
    x0(2) = 1e-3;
  }

  memcpy(lbx0, &x0[0], kNBX0 * sizeof(double));
  memcpy(ubx0, &x0[0], kNBX0 * sizeof(double));

  ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx",
                                lbx0);
  ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx",
                                ubx0);

  /*************************************
  ********* INITIALIZE SOLUTION ********
  **************************************/

  // in our case kNX = kNBX0
  double x_init[kNX];
  memcpy(x_init, lbx0, kNX * sizeof(double));

  double u_init[kNU];
  u_init[kIndAx]    = 0.0;
  u_init[kIndAy]    = 0.0;
  u_init[kIndSDDot] = 0.0;

  // generate params from reference trajectory starting at current s
  double s                             = get_s_from_state(x0);
  std::array<Spline1D, 2> adjusted_ref = compute_adjusted_ref(s);

  std::cout << "[MPCC] starting s is " << s << std::endl;
  std::cout << "[MPCC] adjusted ref is " << adjusted_ref[0](0).coeff(0) << ", "
            << adjusted_ref[1](0).coeff(0) << std::endl;

  // Eigen::Vector2d prev_pos = _prev_x0.head(2);
  Eigen::Vector2d prev_pos = _prev_x0.segment(kNX, 2);
  Eigen::Vector2d curr_pos = x0.head(2);

  double dist = (prev_pos - curr_pos).norm();
  if (_is_shift_warm && dist > 1e-1) {
    std::cout << termcolor::red << "[MPCC] Pos too far (" << dist
              << "), turning off shifted warm start" << std::endl;
    std::cout << "[MPCC] x0: " << x0.transpose() << termcolor::reset
              << std::endl;
    // _is_shift_warm = false;
  }

  double starting_s = _prev_x0[1 * kNX + kIndS];
  if (!_is_shift_warm)
    warm_start_no_u(x_init);
  else {
    // warm_start_shifted_u(false, x0);
    warm_start_shifted_u((prev_pos - curr_pos).norm() > 5e-2, x0);
  }

  /*************************************
  ********* SET REFERENCE PARAMS *******
  **************************************/

  if (!set_solver_parameters(adjusted_ref))
    return {0, 0};

  /*************************************
  ************* RUN SOLVER *************
  **************************************/

  // double elapsed_time = 0.0;
  double timer;

  // run at most 2 times, if first fails, try with simple initialization
  for (int i = 0; i < 2; ++i) {
    // timer for acados using chrono
    auto start = std::chrono::high_resolution_clock::now();
    int status = double_integrator_mpcc_acados_solve(_acados_ocp_capsule);
    auto end   = std::chrono::high_resolution_clock::now();
    // for some reason this causes problems in docker container, commenting
    // out for now
    // ocp_nlp_get(_nlp_config, _nlp_solver, "time_tot", &timer);
    // elapsed_time += timer;

    if (status == ACADOS_SUCCESS) {
      std::cout << "[MPCC] unicycle_model_mpcc_acados_solve(): SUCCESS! "
                << std::chrono::duration<double>(end - start).count() << "s"
                << std::endl;
      //          << elapsed_time * 1000 << std::endl;
      _is_shift_warm = true;
      _solve_success = true;
      break;
    } else {
      _is_shift_warm = false;
      std::cout << termcolor::red
                << "[MPCC] unicycle_model_mpcc_acados_solve() failed with "
                   "status "
                << status << termcolor::reset << std::endl;
      std::cout << "[MPCC] using simple warm start procedure" << std::endl;
      std::cout << "[MPCC] xinit is: " << x0.transpose() << std::endl;
      double h_val = get_cbf_data(x0, Eigen::Vector2d(), true)[0];
      std::cout << "[MPCC] cbf value: " << h_val << std::endl;

      if (h_val < -.1)
        exit(-1);

      warm_start_no_u(x_init);
    }
  }

  /*************************************
  *********** PROCESS OUTPUT ***********
  **************************************/

  double prev_angvel = _prev_u0[kIndS];
  process_solver_output(s);
  std::cout << "mpc x[0] is " << _prev_x0.head(kNX).transpose() << std::endl;
  std::cout << "true x[0] is " << x0.transpose() << std::endl;
  std::cout << "mpc x[1] is " << _prev_x0.segment(kNX, kNX).transpose()
            << std::endl;

  _has_run = true;

  _state << _prev_x0[kIndX], _prev_x0[kIndY], _prev_x0[kIndVx],
      _prev_x0[kIndVy], s, _prev_x0[kIndSDot];

  double new_velx =
      limit(_state[kIndVx], _state[kIndVx] + _prev_u0[kIndAx] * _dt,
            _max_linacc, _dt);
  double new_vely =
      limit(_state[kIndVy], _state[kIndVy] + _prev_u0[kIndAy] * _dt,
            _max_linacc, _dt);

  // ensure velx and y are within input bounds
  new_velx = std::max(std::min(new_velx, _max_linvel), -_max_linvel);
  new_vely = std::max(std::min(new_vely, _max_linvel), -_max_linvel);

  std::cout << "[MPCC] accelerations are: " << _prev_u0[kIndAx] << " "
            << _prev_u0[kIndAy] << std::endl;
  std::cout << "[MPCC] Input is: " << new_velx << " " << new_vely << std::endl;

  _cmd = {new_velx, new_vely};

  return _cmd;
}

void DIMPCC::process_solver_output(double s) {
  // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
  Eigen::VectorXd xtraj((_mpc_steps + 1) * kNX);
  Eigen::VectorXd utraj(_mpc_steps * kNU);
  for (int i = 0; i < _mpc_steps; ++i) {
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "x", &xtraj[i * kNX]);
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "u", &utraj[i * kNU]);
  }

  ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                  &xtraj[_mpc_steps * kNX]);

#ifdef UNICYCLE_MODEL_MPCC_NS
  Eigen::VectorXd slacks(kNS);
  // std::cout << "getting slacks " << NS << std::endl;
  ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, 1, "sl", &slacks[0]);

  std::cout << "[MPCC] Slack values are: " << slacks.transpose() << std::endl;
#endif

  _prev_x0 = xtraj;
  _prev_u0 = utraj;

  for (int i = 0; i <= _mpc_steps; ++i) {
    mpc_x[i]     = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]     = xtraj[kIndY + i * kIndStateInc];
    mpc_vx[i]    = xtraj[kIndVx + i * kIndStateInc];
    mpc_vy[i]    = xtraj[kIndVy + i * kIndStateInc];
    mpc_s[i]     = xtraj[kIndS + i * kIndStateInc] + s;
    mpc_s_dot[i] = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_ax[i]      = utraj[kIndAx + i * kIndInputInc];
    mpc_ay[i]      = utraj[kIndAy + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  xmin << -1e3, -1e3, -_max_linvel, -_max_linvel, 0, -_max_linvel;
  xmax << 1e3, 1e3, _max_linvel, _max_linvel, _ref_length, _max_linvel;
  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  umin << -_max_linacc, -_max_linacc, -_max_linacc;
  umax << _max_linacc, _max_linacc, _max_linacc;

  return {umin, umax};
}
