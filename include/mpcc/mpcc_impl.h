#pragma once

#include <distance_map_core/distance_map_converter_base.h>
#include <distance_map_core/distance_map_converter_instantiater.h>
#include <mpcc/types.h>

#include <Eigen/Core>
#include <map>
#include <vector>

class MPCC {
 public:
  MPCC();
  ~MPCC();

  std::vector<double> solve(const Eigen::VectorXd& state);
  void load_params(const std::map<std::string, double>& params);
  void set_reference(const std::vector<Spline1D>& reference, double arclen);
  void set_reference(const std::vector<SplineWrapper>& reference,
                     double arclen);
  void set_dist_map(const std::shared_ptr<distmap::DistanceMap>& dist_map);
  void set_tubes(const std::vector<SplineWrapper>& tubes);

  Eigen::VectorXd get_state();

  std::vector<double> mpc_x;
  std::vector<double> mpc_y;
  std::vector<double> mpc_theta;
  std::vector<double> mpc_linvels;
  std::vector<double> mpc_s;
  std::vector<double> mpc_s_dot;

  std::vector<double> mpc_angvels;
  std::vector<double> mpc_linaccs;
  std::vector<double> mpc_s_ddots;

 private:
  int _mpc_steps;
  int _x_start;
  int _y_start;
  int _theta_start;
  int _v_start;
  int _s_start;
  int _s_dot_start;
  int _angvel_start;
  int _linacc_start;
  int _s_ddot_start;

  double _bound_value;
  double _max_linvel;
  double _max_angvel;
  double _max_linacc;

  double _alpha;
  double _colinear;
  double _padding;

  double _s_dot;
  double _ref_length;

  bool _use_cbf;
  bool _use_eigen;

  std::shared_ptr<distmap::DistanceMap> _dist_grid_ptr;

  std::map<std::string, double> _params;
  std::vector<Spline1D> _reference;
  std::vector<SplineWrapper> _reference_tk;
  std::vector<SplineWrapper> _tubes;

  Eigen::VectorXd _state;

  double get_s_from_state(const Eigen::VectorXd& state);
};
