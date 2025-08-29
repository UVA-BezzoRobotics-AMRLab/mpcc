#!/usr/bin/env python3

import sys
import yaml
import time
import scipy
import argparse
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from scipy import interpolate
from mpcc_model import (
    export_mpcc_ode_model_spline_param,
    export_mpcc_ode_model_spline_tube_cbf,
    export_mpcc_ode_model_dyna_obs,
)
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

max_s = 8


def create_ocp():

    ocp = AcadosOcp()

    # set model
    # model = export_mpcc_ode_model(list(ss), list(xs), list(ys))
    model = export_mpcc_ode_model_spline_param()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 20

    # Q_mat = 2 * np.diag([1e1, 1e1, 1e-2, 1e-2, 1, 1])
    # R_mat = 2 * 5 * np.diag([1e-1, 1e-3])
    Q_mat = 2 * np.diag([0, 0, 0, 0, 0, 0])
    R_mat = 2 * 5 * np.diag([1e-1, 1e-3])

    # path cost
    # ocp.cost.cost_type = "LINEAR_LS"
    # ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    # ocp.cost.yref = np.zeros((nx + nu,))
    # ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    # ocp.cost.Vx = np.zeros((nx + nu, nx))
    # ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    # ocp.cost.Vu = np.zeros((nx + nu, nu))
    # ocp.cost.Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.cost_type = "EXTERNAL"

    # terminal cost
    ocp.cost.cost_type_e = "EXTERNAL"
    # ocp.cost.cost_type_e = "LINEAR_LS"
    # ocp.cost.yref_e = np.zeros((nx,))
    # ocp.model.cost_y_expr_e = model.x
    # ocp.cost.W_e = Q_mat
    # ocp.cost.Vx_e = np.eye(nx)
    # ns = N

    ocp.dims.N = N
    ocp.parameter_values = np.zeros((nparams,))

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # ocp.cost.zl = 100 * np.ones((ns,))
    # ocp.cost.zu = 100 * np.ones((ns,))
    # ocp.cost.Zl = 1 * np.ones((ns,))
    # ocp.cost.Zu = 1 * np.ones((ns,))

    ocp.constraints.lbu = np.array([-3, -np.pi / 2, -3])
    ocp.constraints.ubu = np.array([3, np.pi / 2, 3])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # theta b/w -pi and pi
    # ocp.constraints.lbx = np.array([-1e6, -1e6, -np.pi, 0, 0, 0])
    # ocp.constraints.ubx = np.array([1e6, 1e6, np.pi, 2, max_s, 2])
    # ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    # theta can be whatever
    ocp.constraints.lbx = np.array([-1e6, -1e6, -1e6, 0, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, 1e6, 2, max_s, 2])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)
    # Partial is slightly slower but more stable allegedly than full condensing.
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # sometimes solver failed due to NaNs, regularizing Hessian helped
    ocp.solver_options.regularize_method = "MIRROR"

    # used these previously and they didn't help anything too much
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


def create_ocp_tube_cbf(yaml_file):

    ocp = AcadosOcp()

    params = None
    if yaml_file != "":
        with open(yaml_file) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("ERROR:", e, file=sys.stderr)
                exit(1)
    else:
        print(
            "ERROR: YAML file must be provided in order to generate MPC code!",
            file=sys.stderr,
        )
        exit(1)

    # set model
    # model = export_mpcc_ode_model(list(ss), list(xs), list(ys))
    model = export_mpcc_ode_model_spline_tube_cbf(params)
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 10

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ocp.dims.nh = 0
    ocp.dims.N = N
    ocp.parameter_values = np.zeros((nparams,))

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    ocp.constraints.lbu = np.array([-3, -np.pi / 2, -3])
    ocp.constraints.ubu = np.array([3, np.pi / 2, 3])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # constraint bounds
    con_upper_bounds = np.array([0])
    con_lower_bounds = np.array([-1e6])

    # hard constraint
    ocp.constraints.uh_0 = con_upper_bounds
    ocp.constraints.lh_0 = con_lower_bounds
    ocp.constraints.uh = con_upper_bounds
    ocp.constraints.lh = con_lower_bounds

    # soft constraint
    ocp.constraints.lsh_0 = np.zeros((1,))
    ocp.constraints.ush_0 = np.zeros((1,))
    ocp.constraints.idxsh_0 = np.array([0])

    ocp.constraints.lsh = np.zeros((1,))
    ocp.constraints.ush = np.zeros((1,))
    ocp.constraints.idxsh = np.array([0])

    # con_upper_bounds = np.array([1e6, 1e6, 0])
    # con_lower_bounds = np.array([0, 0, 1e-6])
    #
    # # hard constraint
    # ocp.constraints.uh_0 = con_upper_bounds
    # ocp.constraints.lh_0 = con_lower_bounds
    # ocp.constraints.uh = con_upper_bounds
    # ocp.constraints.lh = con_lower_bounds
    #
    # # soft constraint
    #
    # ocp.constraints.lsh_0 = np.zeros((1,))
    # ocp.constraints.ush_0 = np.zeros((1,))
    # ocp.constraints.idxsh_0 = np.array([2])
    #
    # ocp.constraints.lsh = np.zeros((1,))
    # ocp.constraints.ush = np.zeros((1,))
    # ocp.constraints.idxsh = np.array([2])

    grad_cost = 100
    hess_cost = 1

    ocp.cost.Zl_0 = hess_cost * np.ones((1,))
    ocp.cost.Zu_0 = hess_cost * np.ones((1,))
    ocp.cost.zl_0 = grad_cost * np.ones((1,))
    ocp.cost.zu_0 = grad_cost * np.ones((1,))

    ocp.cost.Zl = hess_cost * np.ones((1,))
    ocp.cost.Zu = hess_cost * np.ones((1,))
    ocp.cost.zl = grad_cost * np.ones((1,))
    ocp.cost.zu = grad_cost * np.ones((1,))

    # theta can be whatever
    ocp.constraints.lbx = np.array([-1e6, -1e6, -1e6, 0, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, 1e6, 2, max_s, 2])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)

    # Partial is slightly slower but more stable allegedly than full condensing.
    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.hessian_approx = "EXACT"
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    # # sometimes solver failed due to NaNs, regularizing Hessian helped
    # ocp.solver_options.regularize_method = "MIRROR"
    # # ocp.solver_options.tol = 1e-4

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # sometimes solver failed due to NaNs, regularizing Hessian helped
    ocp.solver_options.regularize_method = "MIRROR"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True
    # ocp.solver_options.levenberg_marquardt = 1e-4
    # ocp.solver_options.warm_start_first_qp = 1

    # ocp.solver_options.alpha_min = 0.05  # Default is 0.1, reduce if flickering
    # ocp.solver_options.alpha_reduction = 0.5  # Reduce aggressive steps

    # used these previously and they didn't help anything too much
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


def create_ocp_dyna_obs(yaml_file):

    ocp = AcadosOcp()

    params = None
    if yaml_file != "":
        with open(yaml_file) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("ERROR:", e, file=sys.stderr)
                exit(1)
    else:
        print(
            "ERROR: YAML file must be provided in order to generate MPC code!",
            file=sys.stderr,
        )
        exit(1)

    # set model
    # model = export_mpcc_ode_model(list(ss), list(xs), list(ys))
    model = export_mpcc_ode_model_dyna_obs(params)
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 10

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ocp.dims.nh = 0
    ocp.dims.N = N
    ocp.parameter_values = np.zeros((nparams,))

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    ocp.constraints.lbu = np.array([-3, -np.pi / 2, -3])
    ocp.constraints.ubu = np.array([3, np.pi / 2, 3])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # constraint bounds
    con_upper_bounds = np.array([0, 1e9])
    con_lower_bounds = np.array([-1e6, 0])

    # hard constraint
    ocp.constraints.uh_0 = con_upper_bounds
    ocp.constraints.lh_0 = con_lower_bounds
    ocp.constraints.uh = con_upper_bounds
    ocp.constraints.lh = con_lower_bounds

    # soft constraint
    ocp.constraints.lsh_0 = np.zeros((1,))
    ocp.constraints.ush_0 = np.zeros((1,))
    ocp.constraints.idxsh_0 = np.array([0])

    ocp.constraints.lsh = np.zeros((1,))
    ocp.constraints.ush = np.zeros((1,))
    ocp.constraints.idxsh = np.array([0])

    # con_upper_bounds = np.array([1e6, 1e6, 0])
    # con_lower_bounds = np.array([0, 0, 1e-6])
    #
    # # hard constraint
    # ocp.constraints.uh_0 = con_upper_bounds
    # ocp.constraints.lh_0 = con_lower_bounds
    # ocp.constraints.uh = con_upper_bounds
    # ocp.constraints.lh = con_lower_bounds
    #
    # # soft constraint
    #
    # ocp.constraints.lsh_0 = np.zeros((1,))
    # ocp.constraints.ush_0 = np.zeros((1,))
    # ocp.constraints.idxsh_0 = np.array([2])
    #
    # ocp.constraints.lsh = np.zeros((1,))
    # ocp.constraints.ush = np.zeros((1,))
    # ocp.constraints.idxsh = np.array([2])

    grad_cost = 100
    hess_cost = 1

    ocp.cost.Zl_0 = hess_cost * np.ones((1,))
    ocp.cost.Zu_0 = hess_cost * np.ones((1,))
    ocp.cost.zl_0 = grad_cost * np.ones((1,))
    ocp.cost.zu_0 = grad_cost * np.ones((1,))

    ocp.cost.Zl = hess_cost * np.ones((1,))
    ocp.cost.Zu = hess_cost * np.ones((1,))
    ocp.cost.zl = grad_cost * np.ones((1,))
    ocp.cost.zu = grad_cost * np.ones((1,))

    # theta can be whatever
    # ocp.constraints.lbx = np.array([-1e6, -1e6, -1e6, -0.5, -0.5, -0.5])
    ocp.constraints.lbx = np.array([-1e6, -1e6, -1e6, 0, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, 1e6, 1, max_s, 1])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)

    # Partial is slightly slower but more stable allegedly than full condensing.
    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.hessian_approx = "EXACT"
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    # # sometimes solver failed due to NaNs, regularizing Hessian helped
    # ocp.solver_options.regularize_method = "MIRROR"
    # # ocp.solver_options.tol = 1e-4

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # sometimes solver failed due to NaNs, regularizing Hessian helped
    ocp.solver_options.regularize_method = "MIRROR"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True
    # ocp.solver_options.levenberg_marquardt = 1e-4
    # ocp.solver_options.warm_start_first_qp = 1

    # ocp.solver_options.alpha_min = 0.05  # Default is 0.1, reduce if flickering
    # ocp.solver_options.alpha_reduction = 0.5  # Reduce aggressive steps

    # used these previously and they didn't help anything too much
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


if __name__ == "__main__":
    # ocp = create_ocp()
    # acados_ocp_solver = AcadosOcpSolver(ocp)

    parser = argparse.ArgumentParser(description="test BARN navigation challenge")
    parser.add_argument("--yaml", type=str, default="")

    args = parser.parse_args()

    # ocp = create_ocp_tube_cbf(args.yaml)
    ocp = create_ocp_dyna_obs(args.yaml)
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)
