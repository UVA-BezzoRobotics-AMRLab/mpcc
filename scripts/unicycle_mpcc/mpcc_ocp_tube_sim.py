#!/usr/bin/env python3

import time
import scipy
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from scipy import interpolate
from mpcc_ocp_horizon import create_ocp_tube
from mpcc_model import export_mpcc_ode_model_spline_tube
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import (
    MX,
    vertcat,
    horzcat,
    Function,
    jacobian,
    exp,
    cos,
    sin,
    sqrt,
    bspline,
)

max_s = 4
R = 5


def compute_intermediate_values(
    x1,
    y1,
    theta1,
    v1,
    s1,
    sdot1,
    a,
    w,
    sddot,
    x_coeff,
    y_coeff,
    d_abv_coeff,
    d_blw_coeff,
):
    # Define CasADi symbols
    x1_sym = MX.sym("x1")
    y1_sym = MX.sym("y1")
    theta1_sym = MX.sym("theta1")
    v1_sym = MX.sym("v1")
    s1_sym = MX.sym("s1")
    sdot1_sym = MX.sym("sdot1")
    a_sym = MX.sym("a")
    w_sym = MX.sym("w")
    sddot_sym = MX.sym("sddot")
    x_coeff_sym = MX.sym("x_coeff", 11)
    y_coeff_sym = MX.sym("y_coeff", 11)

    d_abv_coeff_sym = MX.sym("d_abv_coeff", 9)
    d_blw_coeff_sym = MX.sym("d_blw_coeff", 9)

    arc_len_knots = np.linspace(0, 4, 11)
    # arc_len_knots = np.linspace(0, 17.0385372, 11)
    arc_len_knots = np.concatenate(
        (
            np.ones((4,)) * arc_len_knots[0],
            arc_len_knots[2:-2],
            np.ones((4,)) * arc_len_knots[-1],
        )
    )

    v = MX.sym("v")
    x_spline_mx = bspline(v, x_coeff_sym, [list(arc_len_knots)], [3], 1, {})
    y_spline_mx = bspline(v, y_coeff_sym, [list(arc_len_knots)], [3], 1, {})

    spline_x = Function("xr", [v, x_coeff_sym], [x_spline_mx], {})
    spline_y = Function("yr", [v, y_coeff_sym], [y_spline_mx], {})

    xr = spline_x(s1_sym, x_coeff_sym)
    yr = spline_y(s1_sym, y_coeff_sym)

    xr_dot = jacobian(xr, s1_sym)
    yr_dot = jacobian(yr, s1_sym)

    d_abv = 0
    d_blw = 0
    for i in range(9):
        d_abv = d_abv + (d_abv_coeff_sym[i] * s1_sym**i)
        d_blw = d_blw + (d_blw_coeff_sym[i] * s1_sym**i)

    # Define obstacle direction
    obs_dirx = -yr_dot / sqrt(xr_dot**2 + yr_dot**2)
    obs_diry = xr_dot / sqrt(xr_dot**2 + yr_dot**2)

    # Define signed distance
    signed_d = (x1_sym - xr) * obs_dirx + (y1_sym - yr) * obs_diry

    # Define h_abv and h_blw
    p_abv = obs_dirx * cos(theta1_sym) + obs_diry * sin(theta1_sym) + v1_sym * 0.05
    h_abv = (d_abv - signed_d) * exp(-p_abv)

    p_blw = -obs_dirx * cos(theta1_sym) - obs_diry * sin(theta1_sym) + v1_sym * 0.05
    h_blw = (signed_d - d_blw) * exp(-p_blw)

    # Compute h_dot_abv and Lfh_abv
    u = vertcat(a_sym, w_sym, sddot_sym)
    x = vertcat(x1_sym, y1_sym, theta1_sym, v1_sym, s1_sym, sdot1_sym)
    f = vertcat(v1_sym * cos(theta1_sym), v1_sym * sin(theta1_sym), 0, 0, sdot1_sym, 0)
    h_dot_abv = jacobian(h_abv, x)
    Lfh_abv = h_dot_abv @ f

    # Compute h_dot_blw and Lfh_blw
    h_dot_blw = jacobian(h_blw, x)
    Lfh_blw = h_dot_blw @ f
    Lgh_blw = h_dot_blw

    # Define functions to compute these values
    h_abv_fun = Function(
        "h_abv_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [h_abv],
    )
    h_blw_fun = Function(
        "h_blw_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [h_blw],
    )
    h_dot_abv_fun = Function(
        "h_dot_abv_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [h_dot_abv],
    )
    Lfh_abv_fun = Function(
        "Lfh_abv_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [Lfh_abv],
    )
    h_dot_blw_fun = Function(
        "h_dot_blw_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [h_dot_blw],
    )
    Lfh_blw_fun = Function(
        "Lfh_blw_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [Lfh_blw],
    )
    Lgh_blw_fun = Function(
        "Lgh_blw_fun",
        [
            x1_sym,
            y1_sym,
            theta1_sym,
            v1_sym,
            s1_sym,
            sdot1_sym,
            x_coeff_sym,
            y_coeff_sym,
            d_abv_coeff_sym,
            d_blw_coeff_sym,
        ],
        [Lgh_blw],
    )

    # Compute the values
    h_abv_val = h_abv_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    h_blw_val = h_blw_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    h_dot_abv_val = h_dot_abv_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    Lfh_abv_val = Lfh_abv_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    h_dot_blw_val = h_dot_blw_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    Lfh_blw_val = Lfh_blw_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )

    g = vertcat(
        horzcat(0, 0, 0),
        horzcat(0, 0, 0),
        horzcat(0, 1, 0),
        horzcat(1, 0, 0),
        horzcat(0, 0, 0),
        horzcat(0, 0, 1),
    )
    Lgh_blw_val = Lgh_blw_fun(
        x1,
        y1,
        theta1,
        v1,
        s1,
        sdot1,
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
    )
    f_val = vertcat(v1 * cos(theta1), v1 * sin(theta1), 0, 0, sdot1, 0)

    h_dot_val = Lgh_blw_val
    Lgh_blw_val = Lgh_blw_val @ g

    # Print the values
    print("h_abv:", h_abv_val)
    print("h_blw:", h_blw_val)
    print("h_dot_abv:", h_dot_abv_val)
    print("Lfh_abv:", Lfh_abv_val)
    print("h_dot_blw:", h_dot_blw_val)
    print("Lfh_blw:", Lfh_blw_val)

    print("h_dot:", h_dot_val)
    print("f:", f_val)

    const = Lfh_blw_val + Lgh_blw_val @ vertcat(a, w, sddot) + 1.0 * h_blw_val
    print(
        f"cbf lower: {Lfh_blw_val} + {Lgh_blw_val} * [{a}, {w}, {sddot}] + 1.0 * {h_blw_val} = {const}"
    )
    print("v is: ", v1)
    print("sdot is: ", sdot1)


# T_horizon = 2.0  # Define the prediction horizon
def create_reference_circle():

    T = 20
    n = 30

    t = np.linspace(0, T, num=n)
    theta = 2 * np.pi * t / T  # + np.pi

    # circle should be centered at (0,-R)
    # circle should go in the direction from (0,0) to (R, -R) to (0,-2R) to (-R,-R) to (0,0)
    x = R * np.cos(-theta + np.pi / 2)
    y = R * np.sin(-theta + np.pi / 2) - R

    cX = interpolate.CubicSpline(t, x)
    cY = interpolate.CubicSpline(t, y)

    return reparam_curve(cX, cY, T)


def create_reference_line():

    # straight line
    n = 20
    T = 15
    L = 7.0

    # t = np.array([0, .6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 6.6, 7.2, 7.8, 8.4, 9.0, 9.6, 10.2, 10.8, 11.4, 12.0, 12.6, 13.2, 13.8, 14.4])
    t = np.linspace(0, T, num=n)
    # x = np.array([0, .3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0, 6.3, 6.6, 6.9, 7.2])
    x = np.linspace(0, L, num=n)
    y = np.array(n * [0.0])

    # rotate 90 degrees
    theta = 0  # np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for i in range(len(x)):
        [nX, nY] = np.matmul(R, np.array([x[i], y[i]]))
        x[i] = nX
        y[i] = nY

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


def get_footprint(x, y, theta):

    # ellipse parameters
    a = 0.34
    b = 0.31

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Q matrix (before transformation)
    Q = np.diag([a**2, b**2])
    Q_inv = np.linalg.inv(Q)  # Inverse of Q

    # Transformed quadratic form
    A = R @ Q_inv @ R.T  # Equivalent to R * Q^{-1} * R^T

    # Create a grid of points
    x_vals = np.linspace(-2, 6, 400)
    y_vals = np.linspace(-2, 6, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the quadratic form (X - p)^T A (X - p)
    X_shift = X - x
    Y_shift = Y - y
    quad_form = (
        A[0, 0] * X_shift**2 + 2 * A[0, 1] * X_shift * Y_shift + A[1, 1] * Y_shift**2
    )

    # Define rectangle parameters
    rect_center = np.array([0, 0])  # Center of rectangle
    rect_width = 0.5  # Width of rectangle
    rect_height = 0.43  # Height of rectangle
    rect_theta = np.pi / 6  # Rotation angle (same as ellipse for comparison)

    # Compute rectangle corners before rotation
    half_w, half_h = rect_width / 2, rect_height / 2
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]
    ).T  # 2x4 matrix with column-wise corners

    # Rotate and translate rectangle corners
    rot_corners = R @ corners + rect_center[:, np.newaxis]

    # Plot contour at level 1 (which corresponds to the ellipse)
    plt.figure()
    plt.contour(X, Y, quad_form, levels=[1], colors="r")
    plt.scatter([x], [y], color="blue", label="Center (p)")
    plt.plot(
        *np.hstack([rot_corners, rot_corners[:, :1]]),
        "b-",
        linewidth=2,
        label="Rectangle",
    )  # Close rectangle

    plt.legend()
    plt.title("Ellipse from Quadratic Form")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def simulation():

    # get_footprint(0, 0, 0)
    # exit(0)

    yaml_file = "/home/bezzo/catkin_ws/src/mpcc/params/mpcc.yaml"
    ocp = create_ocp_tube(yaml_file)

    ss, xs, ys = create_reference_circle()
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
    Nsim = 150
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
    (footprint,) = ax.plot([], [], "r-", label="Footprint", zorder=1)

    theta = np.linspace(0, 2 * np.pi, 100)

    dist = 0.75
    # inner circle
    in_x = (R - dist) * np.cos(-theta + np.pi / 2)
    in_y = (R - dist) * np.sin(-theta + np.pi / 2) - (R - dist) - dist

    # outer
    out_x = (R + dist) * np.cos(-theta + np.pi / 2)
    out_y = (R + dist) * np.sin(-theta + np.pi / 2) - (R + dist) + dist

    ax.plot(in_x, in_y, "r--", label="inner circ")
    ax.plot(out_x, out_y, "r--", label="outer circ")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Reference vs. Trajectory")

    x_hor_spline = sX
    y_hor_spline = sY

    # k defaults to 3 (cubic)
    bspl_x = interpolate.make_interp_spline(ss, xs)
    bspl_y = interpolate.make_interp_spline(ss, ys)

    model = ocp.model
    constraint_fun = ca.Function(
        "constraint_fun", [model.x, model.u, model.p], [model.con_h_expr]
    )

    samples = np.linspace(0, max_s, 11)
    x_spline_coeff = None
    y_spline_coeff = None

    d_abv_coeffs = None
    d_blw_coeffs = None

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

        else:
            print("AHHHHH WE'RE DONE")
            break

        d_abv_spline = interpolate.make_interp_spline(
            samples, [dist for i in range(11)]
        )
        d_blw_spline = interpolate.make_interp_spline(
            samples, [-dist for i in range(11)]
        )

        for stage in range(N_horizon + 1):

            x_spline_coeff = x_hor_spline.c
            y_spline_coeff = y_hor_spline.c

            d_abv_coeffs = [0] * 9
            d_abv_coeffs[0] = dist
            d_blw_coeffs = [0] * 9
            d_blw_coeffs[0] = -dist

            acados_ocp_solver.set(
                stage,
                "p",
                np.concatenate(
                    (
                        x_spline_coeff,
                        y_spline_coeff,
                        d_abv_coeffs,
                        d_blw_coeffs,
                        [0.0, 100, 0.3, 1.0, 1.0, 1.0, 0.5],
                    )
                    # (x_spline_coeff, y_spline_coeff, d_abv_spline.c, d_blw_spline.c)
                ),
            )

        # time the solver
        start = time.time()
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        # acados_ocp_solver.print_statistics()
        end = time.time()

        print("XCURRENT:", xcurrent)
        print("SIMU:", simU[i, :])
        print(f"{i}/{Nsim} TTS: {(end - start)}")

        sl = acados_ocp_solver.get(1, "sl")
        su = acados_ocp_solver.get(1, "su")
        print("sl", sl, "su", su)

        xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
        simX[i + 1, :] = xcurrent

        x1 = xcurrent[0]
        y1 = xcurrent[1]

        xr = x_hor_spline(0)
        yr = y_hor_spline(0)

        x_hor_spline_d = x_hor_spline.derivative(1)
        y_hor_spline_d = x_hor_spline.derivative(1)

        x_hor_spline_dd = x_hor_spline.derivative(2)
        y_hor_spline_dd = y_hor_spline.derivative(2)

        xr_dot = x_hor_spline_d(0)
        yr_dot = y_hor_spline_d(0)

        xr_ddot = x_hor_spline_dd(0)
        yr_ddot = y_hor_spline_dd(0)

        phi = np.arctan2(yr_dot, xr_dot)

        e_c = np.sin(phi) * (x1 - xr) - np.cos(phi) * (y1 - yr)
        e_l = -np.cos(phi) * (x1 - xr) - np.sin(phi) * (y1 - yr)

        signed_d = ((x1 - xr) * xr_ddot + (y1 - yr) * yr_ddot) / np.sqrt(
            xr_ddot**2 + yr_ddot**2
        )

        print(
            "e_c",
            round(e_c, 2),
            "e_l",
            round(e_l, 2),
            "dist",
            round(np.linalg.norm([e_c, e_l]), 2),
            "signed_d",
            round(signed_d, 2),
            "d_blw_con",
            d_blw_spline(0) - signed_d,
            "d_abv_con",
            signed_d - d_abv_spline(0),
        )
        print("theta", xcurrent[2])

        # print(d_blw_spline.t)
        x_sol = acados_ocp_solver.get(0, "x")
        u_sol = acados_ocp_solver.get(0, "u")
        p_sol = acados_ocp_solver.get(0, "p")
        constraint_val = constraint_fun(x_sol, u_sol, p_sol)
        # if constraint_val[0] < -sl[0] or constraint_val[1] < -sl[1]:
        #     print("constraints violated", constraint_val)
        #     compute_intermediate_values(
        #         x1,
        #         y1,
        #         xcurrent[2],
        #         xcurrent[3],
        #         xcurrent[4],
        #         xcurrent[5],
        #         simU[i, 0],
        #         simU[i, 1],
        #         simU[i, 2],
        #         x_spline_coeff,
        #         y_spline_coeff,
        #         d_abv_coeffs,
        #         d_blw_coeffs,
        #     )
        #     # exit(0)
        #       break

        print("contraint value", constraint_val)

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
