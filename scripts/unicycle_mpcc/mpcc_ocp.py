from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from mpcc_model import export_mpcc_ode_model, export_mpcc_ode_model_spline_param
import numpy as np
import casadi as ca
import scipy
import time

import matplotlib.pyplot as plt

from scipy import interpolate


# T_horizon = 2.0  # Define the prediction horizon
def create_reference():

    # n = 150
    # T = 15
    # L = 7.0

    # t = np.linspace(0, T, num=n)
    # x = np.linspace(0, L, num=n)
    # y = np.array(n * [0.0])

    R = 3
    T = 20
    n = 30

    t = np.linspace(0, T, num=n)
    theta = 2 * np.pi * t / T  # + np.pi

    # circle should be centered at (0,-R)
    # circle should go in the direction from (0,0) to (R, -R) to (0,-2R) to (-R,-R) to (0,0)
    x = R * np.cos(-theta + np.pi / 2)
    y = R * np.sin(-theta + np.pi / 2) - R

    # rotate 90 degrees
    # theta = np.pi / 2
    # theta = 0
    # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # for i in range(len(x)):
    #     [nX, nY] = np.matmul(R, np.array([x[i], y[i]]))
    #     x[i] = nX
    #     y[i] = nY

    cX = interpolate.CubicSpline(t, x, bc_type="clamped")
    cY = interpolate.CubicSpline(t, y, bc_type="clamped")

    return reparam_curve(cX, cY, T)


def compute_arclen(t0, tf, vx, vy):
    dt = (tf - t0) / 10
    ts = np.arange(t0, tf, dt)

    s = 0
    for t in ts:
        s += np.sqrt(vx(t) * vx(t) + vy(t) * vy(t)) * dt

    return s


def binary_search(vx, vy, dl, start, end, tolerance=1e-3):
    t_left = float(start)
    t_right = float(end)

    prev_s = 0
    s = -1000

    while np.abs(prev_s - s) > tolerance:
        prev_s = s

        t_mid = (t_left + t_right) / 2
        s = compute_arclen(float(start), t_mid, vx, vy)

        if s < dl:
            t_left = t_mid
        else:
            t_right = t_mid

    t_mid = round((t_left + t_right) / 2, 6)
    return t_mid


def reparam_curve(cx, cy, tot_t):
    vx = cx.derivative(1)
    vy = cy.derivative(1)

    s = compute_arclen(0, tot_t, vx, vy)

    M = 10
    dl = s / float(M)

    ss = []
    xs = []
    ys = []

    for i in range(M + 1):
        l = i * dl
        ti = binary_search(vx, vy, l, 0, tot_t)
        x = cx(ti)
        y = cy(ti)

        ss.append(l)
        xs.append(x)
        ys.append(y)

    # sX = interpolate.CubicSpline(ss, xs)
    # sY = interpolate.CubicSpline(ss, ys)

    return ss, xs, ys


def create_ocp():

    ocp = AcadosOcp()
    ss, xs, ys = create_reference()

    # set model
    # model = export_mpcc_ode_model(list(ss), list(xs), list(ys))
    model = export_mpcc_ode_model_spline_param()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 10

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

    ocp.constraints.lbx = np.array([-1e6, -1e6, -np.pi, 0, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, np.pi, 2, ss[-1], 2])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


def simulation():

    ocp = create_ocp()

    ss, xs, ys = create_reference()
    sX = interpolate.CubicSpline(ss, xs)
    sY = interpolate.CubicSpline(ss, ys)

    s_space = np.linspace(0, ss[-1], 100)
    x_space = [sX(s) for s in s_space]
    y_space = [sY(s) for s in s_space]

    # model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 200
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    simX[0, :] = xcurrent

    # for stage in range(N_horizon + 1):
    #     acados_ocp_solver.set(stage, "x", xcurrent)
    #     # compute traj_omg
    #     vx = ref[stage][2]
    #     vy = ref[stage][3]
    #     ax = ref[stage][4]
    #     ay = ref[stage][5]

    #     traj_omg = 0
    #     if np.linalg.norm([vx, vy]) > 1e-6:
    #         traj_omg = (ay * vx - ax * vy) / (vx**2 + vy**2)

    #     acados_ocp_solver.set(stage, "p", traj_omg)

    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    fig, ax = plt.subplots()
    # (current_pt,) = ax.plot([], [], "r-", label="Current")
    (line_traj,) = ax.plot([], [], "b-", label="Trajectory")
    ax.plot(x_space, y_space)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Reference vs. Trajectory")

    for i in range(Nsim):

        for stage in range(N_horizon + 1):

            acados_ocp_solver.set(stage, "p", np.zeros([91, 1]))

        # time the solver
        start = time.time()
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        # acados_ocp_solver.print_statistics()
        end = time.time()
        print("XCURRENT:", xcurrent)
        print("SIMU:", simU[i, :])
        print(f"{i}/{Nsim} TTS: {(end - start)}")

        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

        # plt.plot(simX[:, 0], simX[:, 1], "r")
        line_traj.set_xdata(simX[: i + 1, 0])
        line_traj.set_ydata(simX[: i + 1, 1])
        # current_pt.set_data(ref[: i + 1, 0], ref[: i + 1, 1])

        plt.pause(0.01)

    # plot results
    plt.show()


if __name__ == "__main__":
    simulation()
