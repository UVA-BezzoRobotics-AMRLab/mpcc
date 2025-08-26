#!/usr/bin/env python3

import time
import scipy
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from scipy import interpolate
from mpcc_ocp_horizon import create_ocp
from mpcc_model import export_mpcc_ode_model, export_mpcc_ode_model_spline_param
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

max_s = 8


# T_horizon = 2.0  # Define the prediction horizon
def create_reference():

    R = 3
    T = 20
    n = 30

    t = np.linspace(0, T, num=n)
    theta = 2 * np.pi * t / T  # + np.pi

    # circle should be centered at (0,-R)
    # circle should go in the direction from (0,0) to (R, -R) to (0,-2R) to (-R,-R) to (0,0)
    x = R * np.cos(-theta + np.pi / 2)
    y = R * np.sin(-theta + np.pi / 2) - R

    # straight line
    # n = 20
    # T = 15
    # L = 7.0

    # # t = np.array([0, .6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 6.6, 7.2, 7.8, 8.4, 9.0, 9.6, 10.2, 10.8, 11.4, 12.0, 12.6, 13.2, 13.8, 14.4])
    # t = np.linspace(0, T, num=n)
    # # x = np.array([0, .3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0, 6.3, 6.6, 6.9, 7.2])
    # x = np.linspace(0, L, num=n)
    # y = np.array(n * [0.0])

    # # rotate 90 degrees
    # theta = 0  # np.pi / 2
    # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # for i in range(len(x)):
    #     [nX, nY] = np.matmul(R, np.array([x[i], y[i]]))
    #     x[i] = nX
    #     y[i] = nY

    cX = interpolate.CubicSpline(t, x)
    cY = interpolate.CubicSpline(t, y)

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

    # print(ss)
    # print([round(float(x), 4) for x in xs])
    # print([round(float(x), 4) for x in ys])
    # exit(0)
    return ss, xs, ys


def simulation():

    ocp = create_ocp()

    ss, xs, ys = create_reference()
    sX = interpolate.CubicSpline(ss, xs)
    sY = interpolate.CubicSpline(ss, ys)

    s_space = np.linspace(0, ss[-1], 100)
    x_space = [sX(s) for s in s_space]
    y_space = [sY(s) for s in s_space]

    # # model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, build=False)
    acados_integrator = AcadosSimSolver(ocp, build=False)

    N_horizon = acados_ocp_solver.N

    # prepare simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    xcurrent = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    simX[0, :] = xcurrent

    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    fig, ax = plt.subplots()
    # (current_pt,) = ax.plot([], [], "r-", label="Current")
    ax.plot(x_space, y_space, "b--")
    (line_traj,) = ax.plot([], [], "g-", label="Trajectory", zorder=2)
    (horizon,) = ax.plot([], [], "r-", label="Horizon", zorder=3)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Reference vs. Trajectory")

    # knots = np.linspace(0, 2, 11)
    # x_hor_spline = None
    # y_hor_spline = None

    x_hor_spline = sX
    y_hor_spline = sY

    v = ca.MX.sym("v")
    x_coeff = ca.MX.sym("x_coeffs", 11)
    y_coeff = ca.MX.sym("y_coeffs", 11)
    # arc_len_knots = DM([1.0] * 11)
    # arc_len_knots = MX.sym("knots", 11)
    p = ca.vertcat(x_coeff, y_coeff)

    # arc_len_knots = np.linspace(0, 2, 11)
    # arc_len_knots = np.linspace(0, 17.0385372, 11)
    # arc_len_knots = np.concatenate(
    #     (
    #         np.ones((4,)) * arc_len_knots[0],
    #         arc_len_knots[2:-2],
    #         np.ones((4,)) * arc_len_knots[-1],
    #     )
    # )

    # k defaults to 3 (cubic)
    bspl_x = interpolate.make_interp_spline(ss, xs)
    bspl_y = interpolate.make_interp_spline(ss, ys)

    # domain = np.linspace(0, 4, 11)
    # bspl_x = interpolate.make_interp_spline(domain, [bspl_x(x) for x in domain])
    # bspl_y = interpolate.make_interp_spline(domain, [bspl_y(x) for x in domain])

    # 1 denotes the multiplicity of the knots at the ends
    # don't need clamped so leave as 1
    # x_spline_mx = ca.bspline(v, x_coeff, [list(arc_len_knots)], [3], 1, {})
    # y_spline_mx = ca.bspline(v, y_coeff, [list(arc_len_knots)], [3], 1, {})

    # spline_x = ca.Function("xr", [v, x_coeff], [x_spline_mx], {})
    # spline_y = ca.Function("yr", [v, y_coeff], [y_spline_mx], {})

    # print(spline_x(17.01, bspl.c))
    # print(sX(17.01))

    # exit()
    samples = np.linspace(0, max_s, 11)
    for i in range(Nsim):

        # find s minimizing dist to robot
        s = 0
        min_dist = 1e6
        for si in np.linspace(0, ss[-1] - 1, 100):
            d = np.linalg.norm(
                np.array(xcurrent[:2]) - np.array([bspl_x(si), bspl_y(si)])
            )

            if min_dist > d:
                min_dist = d
                s = si

        print("found s", s)
        print("end horizon is", s + max_s)
        print("ss[-1]: ", bspl_x.t[-1])
        xcurrent[4] = 0
        # s = xcurrent[4]
        # for the knots, get the associated cubic spline values
        if s + max_s < bspl_x.t[-1]:
            x_hor = [bspl_x(s + k) for k in samples]
            y_hor = [bspl_y(s + k) for k in samples]

            horizon.set_xdata(x_hor)
            horizon.set_ydata(y_hor)

            x_hor_spline = interpolate.make_interp_spline(samples, x_hor)
            y_hor_spline = interpolate.make_interp_spline(samples, y_hor)

            print(x_hor_spline.t)
            print(x_hor_spline.c)
            print("-------------------")
            print(y_hor_spline.t)
            print(y_hor_spline.c)

            # print(x_hor_spline(0))
            # print(x_hor_spline(1))
            # print(x_hor_spline(2))
            # print(x_hor_spline(3))
            # print(x_hor_spline(4))

        else:
            print("AHHHHH WE'RE DONE")
            break

            # x_hor_spline = interpolate.CubicSpline(knots, x_hor)
            # y_hor_spline = interpolate.CubicSpline(knots, y_hor)

        for stage in range(N_horizon + 1):

            # x_spline_coeff = bspl_x.c
            # y_spline_coeff = bspl_y.c

            x_spline_coeff = x_hor_spline.c
            y_spline_coeff = y_hor_spline.c

            acados_ocp_solver.set(
                stage, "p", np.concatenate((x_spline_coeff, y_spline_coeff))
            )

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

        xcurrent[2] = np.arctan2(np.sin(xcurrent[2]), np.cos(xcurrent[2]))
        print(xcurrent)

        # plt.plot(simX[:, 0], simX[:, 1], "r")
        line_traj.set_xdata(simX[: i + 1, 0])
        line_traj.set_ydata(simX[: i + 1, 1])
        # current_pt.set_data(ref[: i + 1, 0], ref[: i + 1, 1])

        plt.pause(0.01)

    # plot results
    plt.show()


if __name__ == "__main__":
    simulation()
