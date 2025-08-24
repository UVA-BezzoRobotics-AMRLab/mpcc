from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from mpc_cte_model import export_mpcc_ode_model
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

    vX = cX.derivative(1)
    vY = cY.derivative(1)

    aX = cX.derivative(2)
    aY = cY.derivative(2)

    # return the first 2 seconds of the reference as
    # ref[i] = [x, y, v_x, v_y, a_x, a_y]
    # sample every .1 seconds
    num_samples = T * 10
    ref = np.zeros((num_samples, 6))
    for i in range(num_samples):
        ref[i] = [
            cX(i * 0.1),
            cY(i * 0.1),
            vX(i * 0.1),
            vY(i * 0.1),
            aX(i * 0.1),
            aY(i * 0.1),
        ]

    return ref


def create_ocp():

    ocp = AcadosOcp()

    # set model
    model = export_mpcc_ode_model()
    ocp.model = model

    Tf = 2.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparam = model.p.rows()
    N = 20

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    Q_mat = 2 * np.diag([1e1, 1e1, 1e-2, 1e-2, 1, 1])
    R_mat = 2 * 5 * np.diag([1e-1, 1e-3])

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

    ocp.parameter_values = np.zeros((nparam,))

    ocp.constraints.lbu = np.array([-3, -np.pi / 2])
    ocp.constraints.ubu = np.array([3, np.pi / 2])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-1e6, -1e6, -np.pi, -2, -1e6, -1e6])
    ocp.constraints.ubx = np.array([1e6, 1e6, np.pi, 2, 1e6, 1e6])
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 300
    # ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp


def simulation():

    ocp = create_ocp()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    simX[0, :] = xcurrent

    ref = create_reference()

    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
        # compute traj_omg
        vx = ref[stage][2]
        vy = ref[stage][3]
        ax = ref[stage][4]
        ay = ref[stage][5]

        traj_omg = 0
        if np.linalg.norm([vx, vy]) > 1e-6:
            traj_omg = (ay * vx - ax * vy) / (vx**2 + vy**2)

        acados_ocp_solver.set(stage, "p", traj_omg)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    fig, ax = plt.subplots()
    (current_pt,) = ax.plot([], [], "r-", label="Current")
    (line_traj,) = ax.plot([], [], "b-", label="Trajectory")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Reference vs. Trajectory")

    for i in range(Nsim):

        for j in range(N_horizon):

            step = min(len(ref) - 1, i + j)

            # compute yref from ref
            x = ref[step][0]
            y = ref[step][1]
            vx = ref[step][2]
            vy = ref[step][3]
            theta = np.arctan2(vy, vx)
            v = np.sqrt(vx**2 + vy**2)
            cte = 0
            etheta = 0

            acados_ocp_solver.set(
                j,
                "yref",
                np.array([x, y, theta, v, cte, etheta, 0, 0]),
            )

        step = min(len(ref) - 1, i + N_horizon)
        x = ref[step][0]
        y = ref[step][1]
        vx = ref[step][2]
        vy = ref[step][3]
        theta = np.arctan2(vy, vx)
        v = np.sqrt(vx**2 + vy**2)
        cte = 0
        etheta = 0

        acados_ocp_solver.set(
            N_horizon,
            "yref",
            np.array([x, y, theta, v, cte, etheta]),
        )

        # time the solver
        start = time.time()
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        end = time.time()
        print(f"{i}/{Nsim} TTS: {(end - start)*1000}")
        # status = acados_ocp_solver.get_status()

        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

        # plt.plot(simX[:, 0], simX[:, 1], "r")
        line_traj.set_xdata(simX[: i + 1, 0])
        line_traj.set_ydata(simX[: i + 1, 1])
        current_pt.set_data(ref[: i + 1, 0], ref[: i + 1, 1])

        plt.pause(0.1)

    # plot results
    plt.show()


if __name__ == "__main__":
    simulation()
