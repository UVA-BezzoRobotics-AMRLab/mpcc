#include <mpcc/mpcc_nlopt_impl.h>

MPCC::MPCC() {
  // Set default value
  _dt          = .1;
  _mpc_steps   = 10;
  _max_angvel  = 3.0;    // Maximal angvel radian (~30 deg)
  _max_linvel  = 2.0;    // Maximal linvel accel
  _max_linacc  = 4.0;    // Maximal linacc accel
  _bound_value = 1.0e3;  // Bound value for other variables

  _use_cbf  = false;
  _alpha    = 1.0;
  _colinear = 0.01;
  _padding  = .05;

  // managing the indices in this way is far more cache friendly
  _x_start      = 0;
  _y_start      = 1;
  _theta_start  = 2;
  _v_start      = 3;
  _s_start      = 4;
  _s_dot_start  = 5;
  _angvel_start = 6;
  _linacc_start = 7;
  _s_ddot_start = 8;
  _ind_inc      = 9;

  mpc_x.resize(_mpc_steps);
  std::fill(mpc_x.begin(), mpc_x.end(), 0);
  mpc_y.resize(_mpc_steps);
  std::fill(mpc_y.begin(), mpc_y.end(), 0);
  mpc_theta.resize(_mpc_steps);
  std::fill(mpc_theta.begin(), mpc_theta.end(), 0);
  mpc_linvels.resize(_mpc_steps);
  std::fill(mpc_linvels.begin(), mpc_linvels.end(), 0);
  mpc_s.resize(_mpc_steps);
  std::fill(mpc_s.begin(), mpc_s.end(), 0);
  mpc_s_dot.resize(_mpc_steps);
  std::fill(mpc_s_dot.begin(), mpc_s_dot.end(), 0);
  mpc_angvels.resize(_mpc_steps - 1);
  std::fill(mpc_angvels.begin(), mpc_angvels.end(), 0);
  mpc_linaccs.resize(_mpc_steps - 1);
  std::fill(mpc_linaccs.begin(), mpc_linaccs.end(), 0);
  mpc_s_ddots.resize(_mpc_steps - 1);
  std::fill(mpc_s_ddots.begin(), mpc_s_ddots.end(), 0);

  _tubes = {};

  _state = Eigen::VectorXd(6);
  _state << 0, 0, 0, 0, 0, 0;

  // _s_dot is the pseudo-state which dictates how quickly to move the
  // reference
  _s_dot = 0;

  _use_eigen = false;
  iterations = 0;
}

MPCC::~MPCC() {}

void MPCC::load_params(const std::map<std::string, double>& params) {
  _params = params;
  // Init parameters for MPC object
  _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
  _mpc_steps =
      _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
  _max_angvel  = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL")
                                                         : _max_angvel;
  _max_linvel  = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL")
                                                         : _max_linvel;
  _max_linacc  = _params.find("MAX_LINACC") != _params.end()
                     ? _params.at("MAX_LINACC")
                     : _max_linacc;
  _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND")
                                                        : _bound_value;

  _w_angvel   = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL")
                                                        : _w_angvel;
  _w_angvel_d = params.find("W_DANGVEL") != params.end()
                    ? params.at("W_DANGVEL")
                    : _w_angvel_d;
  _w_linvel_d =
      params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

  _use_cbf =
      params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
  _alpha    = params.find("CBF_ALPHA") != params.end() ? params.at("CBF_ALPHA")
                                                       : _alpha;
  _colinear = params.find("CBF_COLINEAR") != params.end()
                  ? params.at("CBF_COLINEAR")
                  : _colinear;
  _padding  = params.find("CBF_PADDING") != params.end()
                  ? params.at("CBF_PADDING")
                  : _padding;

  // std::cerr << "max linvel is " << _max_linvel << std::endl;

  mpc_x.resize(_mpc_steps);
  std::fill(mpc_x.begin(), mpc_x.end(), 0);
  mpc_y.resize(_mpc_steps);
  std::fill(mpc_y.begin(), mpc_y.end(), 0);
  mpc_theta.resize(_mpc_steps);
  std::fill(mpc_theta.begin(), mpc_theta.end(), 0);
  mpc_linvels.resize(_mpc_steps);
  std::fill(mpc_linvels.begin(), mpc_linvels.end(), 0);
  mpc_s.resize(_mpc_steps);
  std::fill(mpc_s.begin(), mpc_s.end(), 0);
  mpc_s_dot.resize(_mpc_steps);
  std::fill(mpc_s_dot.begin(), mpc_s_dot.end(), 0);
  mpc_angvels.resize(_mpc_steps - 1);
  std::fill(mpc_angvels.begin(), mpc_angvels.end(), 0);
  mpc_linaccs.resize(_mpc_steps - 1);
  std::fill(mpc_linaccs.begin(), mpc_linaccs.end(), 0);
  mpc_s_ddots.resize(_mpc_steps - 1);
  std::fill(mpc_s_ddots.begin(), mpc_s_ddots.end(), 0);

  if (_use_cbf)
    std::cout << "[MPCC] Using CBF!" << std::endl;

  std::cout << "\n!! MPC Obj parameters updated !! " << std::endl;
}

void MPCC::set_tubes(const std::vector<SplineWrapper>& tubes) {
  _tubes = tubes;
}

void MPCC::set_reference(const std::vector<Spline1D>& reference,
                         double arclen) {
  _reference  = reference;
  _ref_length = arclen;
  _use_eigen  = true;
}

void MPCC::set_reference(const std::vector<SplineWrapper>& reference,
                         double arclen) {
  _reference_tk = reference;
  _ref_length   = arclen;
  _use_eigen    = false;
}

void MPCC::set_segments(const std::vector<Segment_t>& segments) {
  _segments = segments;
  if (segments.size() != 0)
    _ds = abs(segments[0].s1 - segments[0].s0);
}

double MPCC::get_s_from_state(const Eigen::VectorXd& state) {
  // find the s which minimizes dist to robot
  double s        = 0;
  double min_dist = 1e6;
  Eigen::Vector2d pos(state(0), state(1));
  for (double i = 0.0; i < _ref_length; i += .05) {
    Eigen::Vector2d p;
    if (_use_eigen)
      p = Eigen::Vector2d(_reference[0](i).coeff(0), _reference[1](i).coeff(0));
    else
      p = Eigen::Vector2d(_reference_tk[0].spline(i),
                          _reference_tk[1].spline(i));

    double d = (pos - p).squaredNorm();
    if (d < min_dist) {
      min_dist = d;
      s        = i;
    }
  }

  return s;
}

autodiff::real MPCC::eval_objective(const autodiff::ArrayXreal& x, void* data) {
  MPCC obj = *(MPCC*)data;

  autodiff::real cost = 0.0;

  // std::cout << "OBJECTIVE EVALUATION" << std::endl;

  for (int i = 0; i < obj._mpc_steps; ++i) {
    // find the segment for current s
    // int idx = autodiff::val(x[obj._s_start + obj._ind_inc * i])/obj._ds;
    // Segment_t seg = obj._segments[idx];

    // autodiff::real mx = seg.mx;
    // autodiff::real my = seg.my;
    // autodiff::real bx = seg.bx;
    // autodiff::real by = seg.by;
    // autodiff::real dmx = seg.dmx;
    // autodiff::real dmy = seg.dmy;
    // autodiff::real dbx = seg.dbx;
    // autodiff::real dby = seg.dby;

    // autodiff::real ref_x = mx * x[obj._s_start + obj._ind_inc * i] + bx;
    // autodiff::real ref_y = my * x[obj._s_start + obj._ind_inc * i] + by;

    // autodiff::real ref_vx = dmx * x[obj._s_start + obj._ind_inc * i] +
    // dbx; autodiff::real ref_vy = dmy * x[obj._s_start + obj._ind_inc * i]
    // + dby;

    autodiff::real ref_x = obj._reference_tk[0].spline(
        autodiff::val(x[obj._s_start + obj._ind_inc * i]));
    autodiff::real ref_y = obj._reference_tk[1].spline(
        autodiff::val(x[obj._s_start + obj._ind_inc * i]));

    autodiff::real ref_vx = obj._reference_tk[0].spline.deriv(
        1, autodiff::val(x[obj._s_start + obj._ind_inc * i]));
    autodiff::real ref_vy = obj._reference_tk[1].spline.deriv(
        1, autodiff::val(x[obj._s_start + obj._ind_inc * i]));

    autodiff::real ref_phi = atan2(ref_vy, ref_vx);
    autodiff::real e_c =
        sin(ref_phi) * (x[obj._x_start + obj._ind_inc * i] - ref_x) -
        cos(ref_phi) * (x[obj._y_start + obj._ind_inc * i] - ref_y);
    autodiff::real e_l =
        -cos(ref_phi) * (x[obj._x_start + obj._ind_inc * i] - ref_x) -
        sin(ref_phi) * (x[obj._y_start + obj._ind_inc * i] - ref_y);

    cost += 50 * e_c * e_c;
    cost += 5 * e_l * e_l;
    cost -= x[obj._s_dot_start + obj._ind_inc * i];
  }
  // std::cout << "OBJECTIVE EVALUATION 1" << std::endl;

  // minimize actuator usage
  for (int i = 0; i < obj._mpc_steps - 1; ++i) {
    cost += obj._w_angvel * x[obj._angvel_start + obj._ind_inc * i] *
            x[obj._angvel_start + obj._ind_inc * i];
  }
  // std::cout << "OBJECTIVE EVALUATION 2" << std::endl;

  // regularization
  for (int i = 0; i < obj._mpc_steps - 2; ++i) {
    cost += obj._w_angvel_d *
            (x[obj._angvel_start + obj._ind_inc * (i + 1)] -
             x[obj._angvel_start + obj._ind_inc * i]) *
            (x[obj._angvel_start + obj._ind_inc * (i + 1)] -
             x[obj._angvel_start + obj._ind_inc * i]);
    cost += obj._w_linvel_d *
            (x[obj._linacc_start + obj._ind_inc * (i + 1)] -
             x[obj._linacc_start + obj._ind_inc * i]) *
            (x[obj._linacc_start + obj._ind_inc * (i + 1)] -
             x[obj._linacc_start + obj._ind_inc * i]);
    cost += obj._w_linvel_d *
            (x[obj._s_ddot_start + obj._ind_inc * (i + 1)] -
             x[obj._s_ddot_start + obj._ind_inc * i]) *
            (x[obj._s_ddot_start + obj._ind_inc * (i + 1)] -
             x[obj._s_ddot_start + obj._ind_inc * i]);
  }

  obj.iterations++;

  return cost;
}

double MPCC::objective(const std::vector<double>& x, std::vector<double>& grad,
                       void* data) {
  MPCC obj = *(MPCC*)data;
  autodiff::VectorXreal x_var(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x_var[i] = x[i];
  }

  autodiff::real cost;
  // std::cout << "objective eval" << std::endl;

  if (!grad.empty()) {
    // std::cout << "\twith gradient" << std::endl;
    Eigen::Map<Eigen::VectorXd> grad_map(grad.data(), grad.size());

    autodiff::detail::gradient(obj.eval_objective, autodiff::detail::wrt(x_var),
                               autodiff::at(x_var, data), cost, grad_map);
    // cost = obj.eval_objective(x_var, data);
    // grad_map = autodiff::gradient(cost, x_var);  // Reverse-mode gradient
    // computation
  } else {
    cost = obj.eval_objective(x_var, data);
  }

  return autodiff::val(cost);
}

autodiff::VectorXreal MPCC::eval_constraint(const autodiff::VectorXreal& x,
                                            void* data) {
  MPCC obj = *(MPCC*)data;

  autodiff::VectorXreal result((int)(obj._mpc_steps * obj._state.size()));

  result[obj._x_start]     = x[obj._x_start] - obj._state(0);
  result[obj._y_start]     = x[obj._y_start] - obj._state(1);
  result[obj._theta_start] = x[obj._theta_start] - obj._state(2);
  result[obj._v_start]     = x[obj._v_start] - obj._state(3);
  result[obj._s_start]     = x[obj._s_start] - obj._state(4);
  result[obj._s_dot_start] = x[obj._s_dot_start] - obj._state(5);

  // std::cout << "DYNAMIC CONSTRAINT EVAL" << std::endl;
  for (int i = 0; i < obj._mpc_steps - 1; ++i) {
    autodiff::real x1     = x[obj._x_start + obj._ind_inc * (i + 1)];
    autodiff::real y1     = x[obj._y_start + obj._ind_inc * (i + 1)];
    autodiff::real theta1 = x[obj._theta_start + obj._ind_inc * (i + 1)];
    autodiff::real v1     = x[obj._v_start + obj._ind_inc * (i + 1)];
    autodiff::real s1     = x[obj._s_start + obj._ind_inc * (i + 1)];
    autodiff::real sdot1  = x[obj._s_dot_start + obj._ind_inc * (i + 1)];

    autodiff::real x0     = x[obj._x_start + obj._ind_inc * i];
    autodiff::real y0     = x[obj._y_start + obj._ind_inc * i];
    autodiff::real theta0 = x[obj._theta_start + obj._ind_inc * i];
    autodiff::real v0     = x[obj._v_start + obj._ind_inc * i];
    autodiff::real s0     = x[obj._s_start + obj._ind_inc * i];
    autodiff::real sdot0  = x[obj._s_dot_start + obj._ind_inc * i];

    autodiff::real w0     = x[obj._angvel_start + obj._ind_inc * i];
    autodiff::real a0     = x[obj._linacc_start + obj._ind_inc * i];
    autodiff::real sddot0 = x[obj._s_ddot_start + obj._ind_inc * i];

    // model equations
    // -3 because 3 inputs, result does not include those...
    result[obj._x_start + (obj._ind_inc - 3) * (i + 1)] =
        x1 - (x0 + v0 * cos(theta0) * obj._dt);
    result[obj._y_start + (obj._ind_inc - 3) * (i + 1)] =
        y1 - (y0 + v0 * sin(theta0) * obj._dt);
    result[obj._theta_start + (obj._ind_inc - 3) * (i + 1)] =
        atan2(sin(theta1 - (theta0 + w0 * obj._dt)),
              cos(theta1 - (theta0 + w0 * obj._dt)));
    result[obj._v_start + (obj._ind_inc - 3) * (i + 1)] =
        v1 - (v0 + a0 * obj._dt);
    result[obj._s_start + (obj._ind_inc - 3) * (i + 1)] =
        s1 - (s0 + sdot0 * obj._dt);
    result[obj._s_dot_start + (obj._ind_inc - 3) * (i + 1)] =
        sdot1 - (sdot0 + sddot0 * obj._dt);
    // std::cout << "done doing results i" << std::endl;
  }

  // std::cout << "DONE DYNAMIC CONSTRAINT" << std::endl;
  return result;
}

void MPCC::multi_constraint(unsigned m, double* result, unsigned n,
                            const double* x, double* grad, void* f_data) {
  MPCC obj = *(MPCC*)f_data;

  autodiff::VectorXreal x_real(n);
  for (size_t i = 0; i < n; ++i) {
    x_real(i) = x[i];
  }

  // std::cout << "constraint eval" << std::endl;
  autodiff::VectorXreal F;
  if (grad) {
    // std::cout << "\twith gradient" << std::endl;
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Jx(grad, m, n);
    autodiff::jacobian(obj.eval_constraint, autodiff::detail::wrt(x_real),
                       autodiff::detail::at(x_real, f_data), F, Jx);
  } else
    F = obj.eval_constraint(x_real, f_data);

  for (size_t i = 0; i < m; ++i) {
    result[i] = autodiff::val(F(i));
  }
}

Eigen::VectorXd MPCC::get_state() {
  return _state;
}

std::vector<double> MPCC::solve(const Eigen::VectorXd& state) {
  bool ok = true;

  // if (_use_cbf && _dist_grid_ptr.get() == nullptr)
  // {
  //     std::cerr << "Distance map not yet set" << std::endl;
  //     return {0, 0};
  // }

  if (_use_cbf && _tubes.size() == 0) {
    std::cerr << "Tubes not set yet" << std::endl;
    return {0, 0};
  }

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
    n_constraints += 2 * _mpc_steps;

  const double x     = state[0];
  const double y     = state[1];
  const double theta = state[2];
  const double v     = state[3];

  // pseudo-state variables
  const double s     = get_s_from_state(state);
  const double s_dot = _s_dot;

  std::vector<double> vars_lower_bound(n_vars);
  std::vector<double> vars_upper_bound(n_vars);

  // state bounds
  for (size_t i = 0; i < _mpc_steps; ++i) {
    vars_lower_bound[_x_start + _ind_inc * i]     = -HUGE_VAL;
    vars_lower_bound[_y_start + _ind_inc * i]     = -HUGE_VAL;
    vars_lower_bound[_theta_start + _ind_inc * i] = -HUGE_VAL;
    vars_lower_bound[_v_start + _ind_inc * i]     = -_max_linvel;
    vars_lower_bound[_s_start + _ind_inc * i]     = 0;
    vars_lower_bound[_s_dot_start + _ind_inc * i] = 0;

    vars_upper_bound[_x_start + _ind_inc * i]     = HUGE_VAL;
    vars_upper_bound[_y_start + _ind_inc * i]     = HUGE_VAL;
    vars_upper_bound[_theta_start + _ind_inc * i] = HUGE_VAL;
    vars_upper_bound[_v_start + _ind_inc * i]     = _max_linvel;
    vars_upper_bound[_s_start + _ind_inc * i]     = _ref_length;
    vars_upper_bound[_s_dot_start + _ind_inc * i] = _max_linvel;
  }

  for (size_t i = 0; i < _mpc_steps - 1; ++i) {
    vars_lower_bound[_angvel_start + _ind_inc * i] = -_max_angvel;
    vars_lower_bound[_linacc_start + _ind_inc * i] = -_max_linacc;
    vars_lower_bound[_s_ddot_start + _ind_inc * i] = -_max_linacc;

    vars_upper_bound[_angvel_start + _ind_inc * i] = _max_angvel;
    vars_upper_bound[_linacc_start + _ind_inc * i] = _max_linacc;
    vars_upper_bound[_s_ddot_start + _ind_inc * i] = _max_linacc;
  }

  nlopt::opt opt(nlopt::algorithm::LD_SLSQP, n_vars);

  opt.set_lower_bounds(vars_lower_bound);
  opt.set_upper_bounds(vars_upper_bound);

  opt.set_min_objective(objective, this);

  // no use using cbf if distance is large
  // double d = dist_grid_ptr->atPositionSafe(x, y, true);
  // if (use_cbf && dist_grid_ptr.get() != nullptr && d < 100)
  // {
  //     std::vector<double> barrier_tolerances(n_barrier_constraints, 1e-8);
  //     opt.add_inequality_mconstraint(inequality_constraint, this,
  //     barrier_tolerances);
  // }

  std::vector<double> dynamics_tolerances(n_constraints, 1e-8);
  opt.add_equality_mconstraint(multi_constraint, this, dynamics_tolerances);

  opt.set_xtol_rel(1e-2);
  // opt.set_ftol_abs(1e-3);
  opt.set_maxtime(autodiff::val(_dt) * .9);

  std::vector<double> x0(n_vars);

  // warm start with previous solution if available
  if (prev_x0.size() != 0) {
    for (size_t i = 0; i < n_vars; ++i) {
      x0[i] = prev_x0[i];
    }
  }

  x0[_x_start]     = x;
  x0[_y_start]     = y;
  x0[_theta_start] = theta;
  x0[_v_start]     = v;
  x0[_s_start]     = s;
  x0[_s_dot_start] = s_dot;

  _state << x0[_x_start], x0[_y_start], x0[_theta_start], x0[_v_start],
      x0[_s_start], x0[_s_dot_start];

  double minf;
  std::vector<double> x0_cp = x0;
  nlopt::result result;
  iterations = 0;
  try {
    result = opt.optimize(x0, minf);
    std::cout << "nlopt succeeded in " << iterations << " iterations"
              << std::endl;
    // std::cout << "cost is " << minf << std::endl;
    // std::cout << "states" << std::endl;
    // for (int i = 0; i < _mpc_steps; ++i)
    // {
    //     std::cout << i << "\tx: " << x0[_x_start + i] << "\ty: " <<
    //     x0[_y_start + i] << "\ttheta: " << x0[_theta_start + i] << "\tv: "
    //     << x0[_v_start + i] << std::endl;
    // }
  } catch (std::exception& e) {
    std::cerr << "nlopt failed: " << e.what() << std::endl;
    // exit(0);
    return {0, 0};
  }

  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_x[i]       = x0[_x_start + _ind_inc * i];
    mpc_y[i]       = x0[_y_start + _ind_inc * i];
    mpc_theta[i]   = x0[_theta_start + _ind_inc * i];
    mpc_linvels[i] = x0[_v_start + _ind_inc * i];
    mpc_s[i]       = x0[_s_start + _ind_inc * i];
    mpc_s_dot[i]   = x0[_s_dot_start + _ind_inc * i];
  }

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    mpc_angvels[i] = x0[_angvel_start + _ind_inc * i];
    mpc_linaccs[i] = x0[_linacc_start + _ind_inc * i];
    mpc_s_ddots[i] = x0[_s_ddot_start + _ind_inc * i];
  }

  // std::cout << "MPC INPUTS ARE: " << std::endl;
  // std::cout << "acc: " << solution.x[_linacc_start] << std::endl;
  // std::cout << "anvel: " << solution.x[_angvel_start] << std::endl;
  // std::cout << "s ddot: " << solution.x[_s_ddot_start] << std::endl;

  _s_dot = x0[_s_dot_start + _ind_inc];

  std::vector<double> input;
  input.push_back(x0[_angvel_start]);
  input.push_back(x0[_linacc_start]);

  return input;
}
