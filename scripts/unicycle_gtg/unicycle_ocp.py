from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from unicycle_model import export_unicycle_ode_model
import numpy as np
import casadi as ca
from utils import plot_robot
import scipy

from scipy import interpolate


def create_ocp():

    ocp = AcadosOcp()

    # set model
    model = export_unicycle_ode_model()
    ocp.model = model

    Tf = 2.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 20

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    Q_mat = 2 * np.diag([1, 1, 1e-1, 1e-1])
    R_mat = 2 * 5 * np.diag([1e-1, 1e-2])
    # Q_mat = np.eye(nx)
    # R_mat = np.eye(nu)

    # initial cost
    # ocp.cost.cost_type_0 = "LINEAR_LS"
    # ocp.model.cost_y_expr_0 = ca.vertcat(model.x, model.u)
    # ocp.cost.yref_0 = np.zeros((nx + nu,))
    # ocp.cost.W_0 = ca.diagcat(Q_mat, R_mat).full()

    # set Vx_0 to (nx+nu) x (nx) matrix
    # ocp.cost.Vx_0 = np.zeros((nx + nu, nx))
    # ocp.cost.Vx_0[:nx, :nx] = np.eye(nx)
    # ocp.cost.Vu_0 = np.zeros((nx + nu, nu))
    # ocp.cost.Vu_0[nx:, :] = np.eye(nu)

    # path cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.zeros((nx + nu,))
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.Vx = np.zeros((nx + nu, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((nx + nu, nu))
    ocp.cost.Vu[nx : nx + nu, 0:nu] = np.eye(nu)

    # terminal cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.yref_e = np.zeros((nx,))
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat
    ocp.cost.Vx_e = np.eye(nx)

    ocp.constraints.lbu = np.array([-3, -np.pi / 2])
    ocp.constraints.ubu = np.array([3, np.pi / 2])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-1e6, -1e6, -np.pi, -2])
    ocp.constraints.ubx = np.array([1e6, 1e6, np.pi, 2])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 200
    # ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


def simulation():

    ocp = create_ocp()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = np.array([0.0, 0.0, 0.0, 0.0])  # Intital state
    simX[0, :] = xcurrent

    yref = np.array([1, 1, 0, 0, 0, 0])
    yref_N = np.array([1, 1, 0, 0])

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # closed loop
    for i in range(Nsim):
        # update yref
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", yref)
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        status = acados_ocp_solver.get_status()

        # if status not in [0, 2]:
        #     acados_ocp_solver.print_statistics()
        #     plot_robot(
        #         np.linspace(0, T_horizon / N_horizon * i, i + 1),
        #         F_max,
        #         simU[:i, :],
        #         simX[: i + 1, :],
        #     )
        #     raise Exception(
        #         f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        #     )

        # simulate system
        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

    # # plot results
    # plot_robot(
    #     np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1),
    #     [F_max, None],
    #     simU,
    #     simX,
    #     x_labels=model.x_labels,
    #     u_labels=model.u_labels,
    #     time_label=model.t_label,
    # )
    print(simX)


if __name__ == "__main__":
    simulation()
