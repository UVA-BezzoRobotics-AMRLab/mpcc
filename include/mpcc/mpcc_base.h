#pragma once

#include <mpcc/types.h>
#include <mpcc/utils.h>

#include <Eigen/Core>
//
// acados
#include "acados_c/ocp_nlp_interface.h"

#include <cmath>
#include <map>

enum class CommandOrder { kPos = 0, kVel, kAccel };

class Command {
 public:
  virtual ~Command() = default;

  Command(CommandOrder order) : _order(order) {}

  virtual void setCommand(double cmd1, double cmd2) = 0;

  virtual std::array<double, 2> getCommand() const = 0;

  CommandOrder getOrder() const { return _order; }

 protected:
  CommandOrder _order = CommandOrder::kVel;
};

// Interface assumes the use of acados
class MPCBase {

 public:
  virtual ~MPCBase() = default;
  virtual void load_params(const std::map<std::string, double>& params) = 0;

  virtual std::array<double, 2> solve(const Eigen::VectorXd& state,
                                      bool is_reverse = false) = 0;

  virtual void set_odom(const Eigen::VectorXd& odom) = 0;
  virtual void reset_horizon()                       = 0;

  void set_tubes(const std::array<Eigen::VectorXd, 2>& tubes) {
    _tubes = tubes;
  }

  void set_reference(const std::array<Spline1D, 2>& reference, double arclen) {
    _reference  = reference;
    _ref_length = arclen;
    return;
  }

  virtual const Eigen::VectorXd& get_state() const                      = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits() const = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits() const = 0;

  virtual std::array<double, 2> get_command() const { return _cmd; }
  virtual Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control,
                                       bool is_abv) const = 0;

  const bool get_solver_status() const { return _solve_success; }

  virtual std::array<Spline1D, 2> compute_adjusted_ref(double s) const {
    // get reference for next _ref_len_sz meters, indexing from s=0 onwards
    // need to also down sample the tubes
    Eigen::RowVectorXd ss, xs, ys;  //, abvs, blws;
    ss.resize(_ref_samples);
    xs.resize(_ref_samples);
    ys.resize(_ref_samples);

    double px = _reference[0](_ref_length).coeff(0);
    double py = _reference[1](_ref_length).coeff(0);

    // capture reference at each sample
    for (int i = 0; i < _ref_samples; ++i) {
      ss(i) = ((double)i) * _ref_len_sz / (_ref_samples - 1);

      // if sample domain exceeds trajectory, duplicate final point
      if (ss(i) + s <= _ref_length) {
        xs(i) = _reference[0](ss(i) + s).coeff(0);
        ys(i) = _reference[1](ss(i) + s).coeff(0);
      } else {
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

  double limit(double prev_val, double input, double max_rate,
               double dt) const {
    double ret = input;
    if (fabs(prev_val - input) / dt > max_rate) {
      if (input > prev_val)
        ret = prev_val + max_rate * dt;
      else
        ret = prev_val - max_rate * dt;
    }

    return ret;
  }

 protected:
  sim_config* _sim_config;
  sim_in* _sim_in;
  sim_out* _sim_out;
  void* _sim_dims;

  ocp_nlp_in* _nlp_in;
  ocp_nlp_out* _nlp_out;
  ocp_nlp_dims* _nlp_dims;
  ocp_nlp_config* _nlp_config;
  ocp_nlp_solver* _nlp_solver;
  void* _nlp_opts;

  double* _new_time_steps;

  Eigen::VectorXd _state;
  Eigen::VectorXd _odom;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  std::map<std::string, double> _params;

  std::array<double, 2> _cmd;
  std::array<Spline1D, 2> _reference;
  std::array<Eigen::VectorXd, 2> _tubes;

  int _mpc_steps;
  int _ref_samples;

  double _dt;
  double _ref_length;
  double _ref_len_sz;

  bool _solve_success;
};
