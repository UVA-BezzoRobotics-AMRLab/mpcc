#pragma once

#include <mpcc/mpcc_base.h>
#include <cstdlib>

#include <Eigen/Dense>
#include <map>
#include <vector>

// acados
#include "acados_sim_solver_unicycle_model_mpcc.h"
#include "acados_solver_unicycle_model_mpcc.h"

class UnicycleCommand : public Command {
 public:
  UnicycleCommand() = default;

  UnicycleCommand(CommandOrder order, double cmd1, double cmd2)
      : Command(order) {
    setCommand(cmd1, cmd2);
  }

  virtual void setCommand(double cmd1, double cmd2) override {
    switch (_order) {
      case CommandOrder::kPos:
        throw std::invalid_argument(
            "[Command] pos cmd not implemented for unicycle.");
      case CommandOrder::kVel:
        _v = cmd1;
        _w = cmd2;
      case CommandOrder::kAccel:
        _a = cmd1;
        _w = cmd2;
        break;
      default:
        throw std::invalid_argument(
            "[Command] invalid order for unicycle command.");
    }
  }

  virtual std::array<double, 2> getCommand() const {

    switch (_order) {
      case CommandOrder::kPos:
        throw std::invalid_argument(
            "[Command] pos cmd not implemented for unicycle.");
      case CommandOrder::kVel:
        return {_v, _w};
      case CommandOrder::kAccel:
        return {_a, _w};
      default:
        throw std::invalid_argument(
            "[Command] invalid order for unicycle command.");
    }
  }

 private:
  double _v = 0;
  double _a = 0;
  double _w = 0;

  CommandOrder _order = CommandOrder::kAccel;
};

class MPCC : public MPCBase {
 public:
  MPCC();
  virtual ~MPCC() override;

  virtual std::array<double, 2> solve(const Eigen::VectorXd& state,
                                      bool is_reverse = false) override;

  virtual void load_params(
      const std::map<std::string, double>& params) override;
  /**********************************************************************
   * Function: MPCC::load_params()
   * Description: Loads parameters for the MPC controller
   * Parameters:
   * @param params: const std::map<std::string, double>&
   * Returns:
   * N/A
   * Notes:
   * This function loads parameters for the MPC controller, including
   * the time step, maximum angular and body rates, weights for the
   * mpcc cost function, and CBF/CLF parameters
   **********************************************************************/

  /***********************
   * Setters and Getters
   ***********************/
  void reset_horizon() override;
  virtual void set_odom(const Eigen::VectorXd& odom) override;
  void set_dyna_obs(const Eigen::MatrixXd& dyna_obs);

  Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               bool is_abv) const override;

  virtual const Eigen::VectorXd& get_state() const override { return _state; };
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits()
      const override;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits()
      const override;

 public:
  // TOOD: make getter for these
  // Use one vector which stores a state struct...
  std::vector<double> mpc_x;
  std::vector<double> mpc_y;
  std::vector<double> mpc_theta;
  std::vector<double> mpc_linvels;
  std::vector<double> mpc_s;
  std::vector<double> mpc_s_dot;

  std::vector<double> mpc_angvels;
  std::vector<double> mpc_linaccs;
  std::vector<double> mpc_s_ddots;

  static constexpr uint16_t kNX = UNICYCLE_MODEL_MPCC_NX;
#ifdef UNICYCLE_MODEL_MPCC_NS
  static constexpr uint16_t kNS = UNICYCLE_MODEL_MPCC_NS;
#endif
  static constexpr uint16_t kNP   = UNICYCLE_MODEL_MPCC_NP;
  static constexpr uint16_t kNU   = UNICYCLE_MODEL_MPCC_NU;
  static constexpr uint16_t kNBX0 = UNICYCLE_MODEL_MPCC_NBX0;

 private:
  /*std::array<Spline1D, 2> compute_adjusted_ref(double s) const;*/
  /**********************************************************************
   * Function: MPCC::get_ref_from_s()
   * Description: Generates a reference trajectory from a given arc length
   * Parameters:
   * @param s: double
   * Returns:
   * a reparameterized trajectory starting at arc length s=0
   * Notes:
   * If the trajectory is shorter than required mpc size, then the
   * last point is repeated for spline generation.
   **********************************************************************/

  double get_s_from_state(const Eigen::VectorXd& state);
  /**********************************************************************
   * Function: MPCC::get_s_from_state()
   * Description: Get the arc length of closest point on reference trajectory
   * Parameters:
   * @param state: const Eigen::VectorXd&
   * Returns:
   * arc length value of closest point to state
   **********************************************************************/

  Eigen::VectorXd next_state(const Eigen::VectorXd& current_state,
                             const Eigen::VectorXd& control);
  /**********************************************************************
   * Function: MPCC::next_state()
   * Description: Calculates the next state of the robot given current
   * state and control input
   * Parameters:
   * @param current_state: const Eigen::VectorXd&
   * @param control: const Eigen::VectorXd&
   * Returns:
   * Next state of the robot
   **********************************************************************/

  void warm_start_no_u(double* x_init);
  /**********************************************************************
   * Function: MPCC::warm_start_no_u()
   * Description: Warm starts the MPC solver with no control inputs
   * Parameters:
   * @param x_init: double*
   * Returns:
   * N/A
   * Notes:
   * This function sets the initial state for the MPC solver assuming
   * a 0 control input
   **********************************************************************/

  void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd& state);
  /**********************************************************************
   * Function: MPCC::warm_start_shifted_u()
   * Description: Warm starts the MPC solver with shifted control inputs
   * Parameters:
   * @param correct_perturb: bool
   * @param state: const Eigen::VectorXd&
   * Returns:
   * N/A
   * Notes:
   * This function sets the initial state for the MPC solver by shifting
   * the control inputs and states from the previous solution.
   * See From linear to nonlinear MPC: bridging the gap via the
   * real-time iteration, Gros et. al. for more details.
   **********************************************************************/

  void process_solver_output(double s);
  /**********************************************************************
   * Function: MPCC::process_solver_output()
   * Description: Processes the output of the MPC solver
   * Parameters:
   * @param s: double
   * Returns:
   * N/A
   **********************************************************************/

  bool set_solver_parameters(const std::array<Spline1D, 2>& adjusted_ref);
  /**********************************************************************
   * Function: MPCC::set_solver_parameters()
   * Description: Sets the parameters for the MPC solver
   * Parameters:
   * @param ref: const std::vector<Spline1D>&
   * Returns:
   * bool - true if successful, false otherwise
   **********************************************************************/

 private:
  static constexpr uint8_t kIndX        = 0;
  static constexpr uint8_t kIndY        = 1;
  static constexpr uint8_t kIndTheta    = 2;
  static constexpr uint8_t kIndV        = 3;
  static constexpr uint8_t kIndS        = 4;
  static constexpr uint8_t kIndSDot     = 5;
  static constexpr uint8_t kIndStateInc = 6;

  static constexpr uint8_t kIndLinAcc   = 0;
  static constexpr uint8_t kIndAngVel   = 1;
  static constexpr uint8_t kIndSDDot    = 2;
  static constexpr uint8_t kIndInputInc = 3;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  Eigen::MatrixXd _dyna_obs;

  double _ds;

  double _bound_value;
  double _max_linvel;
  double _max_angvel;
  double _max_linacc;
  double _max_anga;

  double _alpha_abv;
  double _alpha_blw;
  double _colinear;
  double _padding;

  double _s_dot;

  double _gamma;
  double _w_ql_lyap;
  double _w_qc_lyap;

  double _w_angvel;
  double _w_angvel_d;
  double _w_linvel_d;
  double _w_ql;
  double _w_qc;
  double _w_q_speed;

  unsigned int iterations;

  bool _use_cbf;
  bool _use_eigen;
  bool _is_shift_warm;
  bool _use_dyna_obs;
  bool _has_run;

  unicycle_model_mpcc_sim_solver_capsule* _acados_sim_capsule;
  unicycle_model_mpcc_solver_capsule* _acados_ocp_capsule;
};
