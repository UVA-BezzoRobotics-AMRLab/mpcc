#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

// acados
#include "acados_c/ocp_nlp_interface.h"

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
  virtual void load_params(const std::map<std::string, double> &params) = 0;

  virtual Command &solve(const Eigen::VectorXd &state,
                         bool is_reverse = false) = 0;

  virtual void set_odom(const Eigen::VectorXd &odom) = 0;

  virtual Command &get_command() const = 0;

  double limit(double prev_val, double input, double max_rate) const {
    double ret = input;
    if (fabs(prev_val - input) / _dt > max_rate) {
      if (input > prev_val)
        ret = prev_val + max_rate * _dt;
      else
        ret = prev_val - max_rate * _dt;
    }

    return ret;
  }

protected:
  sim_config *_sim_config;
  sim_in *_sim_in;
  sim_out *_sim_out;
  void *_sim_dims;

  ocp_nlp_in *_nlp_in;
  ocp_nlp_out *_nlp_out;
  ocp_nlp_dims *_nlp_dims;
  ocp_nlp_config *_nlp_config;
  ocp_nlp_solver *_nlp_solver;
  void *_nlp_opts;

  Eigen::VectorXd _state;
  Eigen::VectorXd _odom;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  int _mpc_steps;

  double _dt;
};
