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
from mpcc_model import create_mpcc_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

max_s = 4

def create_ocp(yaml_file):
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
    model = create_mpcc_ode_model()
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

    # ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    # ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    # ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # grad_cost = 100
    # hess_cost = 1
    #
    # ocp.cost.Zl_0 = hess_cost * np.ones((1,))
    # ocp.cost.Zu_0 = hess_cost * np.ones((1,))
    # ocp.cost.zl_0 = grad_cost * np.ones((1,))
    # ocp.cost.zu_0 = grad_cost * np.ones((1,))
    #
    # ocp.cost.Zl = hess_cost * np.ones((1,))
    # ocp.cost.Zu = hess_cost * np.ones((1,))
    # ocp.cost.zl = grad_cost * np.ones((1,))
    # ocp.cost.zu = grad_cost * np.ones((1,))

    # theta can be whatever
    ocp.constraints.lbx = np.array([-1e6, -1e6, -4, -4, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, 4, 4, max_s, 4])
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
    parser = argparse.ArgumentParser(description="Double integrator mpcc")
    parser.add_argument("--yaml", type=str, default="")

    args = parser.parse_args()

    # ocp = create_ocp_tube_cbf(args.yaml)
    ocp = create_ocp(args.yaml)
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

