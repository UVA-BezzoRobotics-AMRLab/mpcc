import numpy as np
from acados_template import AcadosModel
from casadi import (
    MX,
    vertcat,
    horzcat,
    sin,
    cos,
    atan2,
    sqrt,
    exp,
    jacobian,
    interpolant,
    bspline,
    Function,
    DM,
)


def export_mpcc_ode_model_spline_param() -> AcadosModel:

    model_name = "unicycle_model_mpcc"

    # set up states & controls
    x1 = MX.sym("x1")
    y1 = MX.sym("y1")
    theta1 = MX.sym("theta1")
    v1 = MX.sym("v1")
    s1 = MX.sym("s1")
    sdot1 = MX.sym("sdot1")

    x = vertcat(x1, y1, theta1, v1, s1, sdot1)

    a = MX.sym("a")
    w = MX.sym("w")
    sddot = MX.sym("sddot")

    u = vertcat(a, w, sddot)

    v = MX.sym("v")
    x_coeff = MX.sym("x_coeffs", 11)
    y_coeff = MX.sym("y_coeffs", 11)
    # arc_len_knots = DM([1.0] * 11)
    # arc_len_knots = MX.sym("knots", 11)
    p = vertcat(x_coeff, y_coeff)

    arc_len_knots = np.linspace(0, 4, 11)
    # arc_len_knots = np.linspace(0, 17.0385372, 11)
    arc_len_knots = np.concatenate(
        (
            np.ones((4,)) * arc_len_knots[0],
            arc_len_knots[2:-2],
            np.ones((4,)) * arc_len_knots[-1],
        )
    )

    # 1 denotes the multiplicity of the knots at the ends
    # don't need clamped so leave as 1
    x_spline_mx = bspline(v, x_coeff, [list(arc_len_knots)], [3], 1, {})
    y_spline_mx = bspline(v, y_coeff, [list(arc_len_knots)], [3], 1, {})

    spline_x = Function("xr", [v, x_coeff], [x_spline_mx], {})
    spline_y = Function("yr", [v, y_coeff], [y_spline_mx], {})

    xr = spline_x(s1, x_coeff)
    yr = spline_y(s1, y_coeff)

    xr_dot = jacobian(xr, s1)
    yr_dot = jacobian(yr, s1)

    phi_r = atan2(xr_dot, yr_dot)

    e_c = sin(phi_r) * (x1 - xr) - cos(phi_r) * (y1 - yr)
    e_l = -cos(phi_r) * (x1 - xr) - sin(phi_r) * (y1 - yr)

    Q_c = 4.0  # 50
    Q_l = 100  # 3
    Q_mat = np.diag([Q_c, Q_l, 1e-1, 5e-1, 1e-1])
    Q_mat_e = np.diag([Q_c, Q_l])  # / 10

    y_expr = vertcat(e_c, e_l, a, w, sddot)
    y_expr_e = vertcat(e_c, e_l)

    # xdot
    x1_dot = MX.sym("x1_dot")
    y1_dot = MX.sym("y1_dot")
    theta1_dot = MX.sym("theta1_dot")
    v1_dot = MX.sym("v1_dot")
    s1_dot = MX.sym("s1_dot")
    sdot1_dot = MX.sym("sdot1_dot")

    xdot = vertcat(x1_dot, y1_dot, theta1_dot, v1_dot, s1_dot, sdot1_dot)

    # dynamics
    cos_theta = cos(theta1)
    sin_theta = sin(theta1)
    f_expl = vertcat(
        v1 * cos_theta,
        v1 * sin_theta,
        w,
        a,
        sdot1,
        sddot,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.name = model_name

    model.cost_expr_ext_cost = y_expr.T @ Q_mat @ y_expr - 0.2 * sdot1
    model.cost_expr_ext_cost_e = y_expr_e.T @ Q_mat_e @ y_expr_e - 0.2 * sdot1

    # store meta information
    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        r"$\theta$ [rad]",
        "$v$ [m]",
        "$s$ []",
        "$sdot$ []",
    ]
    model.u_labels = ["$a$", "$w$", "$sddot$"]
    model.t_label = "$t$ [s]"

    return model


def export_mpcc_ode_model_spline_tube(params) -> AcadosModel:

    model_name = "unicycle_model_mpcc"

    # set up states & controls
    x1 = MX.sym("x1")
    y1 = MX.sym("y1")
    theta1 = MX.sym("theta1")
    v1 = MX.sym("v1")
    s1 = MX.sym("s1")
    sdot1 = MX.sym("sdot1")

    x = vertcat(x1, y1, theta1, v1, s1, sdot1)

    a = MX.sym("a")
    w = MX.sym("w")
    sddot = MX.sym("sddot")

    u = vertcat(a, w, sddot)

    v = MX.sym("v")
    x_coeff = MX.sym("x_coeffs", params["mpc_ref_samples"])
    y_coeff = MX.sym("y_coeffs", params["mpc_ref_samples"])
    # d_abv_coeff = MX.sym("d_above_coeffs", 11)
    # d_blw_coeff = MX.sym("d_below_coeffs", 11)
    d_abv_coeff = MX.sym("d_above_coeffs", params["tube_poly_degree"] + 1)
    d_blw_coeff = MX.sym("d_below_coeffs", params["tube_poly_degree"] + 1)

    arc_len_knots = np.linspace(0, params["ref_length_size"], params["mpc_ref_samples"])
    # arc_len_knots = np.linspace(0, 17.0385372, 11)
    arc_len_knots = np.concatenate(
        (
            np.ones((4,)) * arc_len_knots[0],
            arc_len_knots[2:-2],
            np.ones((4,)) * arc_len_knots[-1],
        )
    )

    # 1 denotes the multiplicity of the knots at the ends
    # don't need clamped so leave as 1
    x_spline_mx = bspline(v, x_coeff, [list(arc_len_knots)], [3], 1, {})
    y_spline_mx = bspline(v, y_coeff, [list(arc_len_knots)], [3], 1, {})

    d_abv = 0
    d_blw = 0
    for i in range(params["tube_poly_degree"] + 1):
        d_abv = d_abv + (d_abv_coeff[i] * s1**i)
        d_blw = d_blw + (d_blw_coeff[i] * s1**i)

    spline_x = Function("xr", [v, x_coeff], [x_spline_mx], {})
    spline_y = Function("yr", [v, y_coeff], [y_spline_mx], {})

    xr = spline_x(s1, x_coeff)
    yr = spline_y(s1, y_coeff)

    xr_dot = jacobian(xr, s1)
    yr_dot = jacobian(yr, s1)

    phi_r = atan2(xr_dot, yr_dot)

    e_c = sin(phi_r) * (x1 - xr) - cos(phi_r) * (y1 - yr)
    e_l = -cos(phi_r) * (x1 - xr) - sin(phi_r) * (y1 - yr)

    # xdot
    x1_dot = MX.sym("x1_dot")
    y1_dot = MX.sym("y1_dot")
    theta1_dot = MX.sym("theta1_dot")
    v1_dot = MX.sym("v1_dot")
    s1_dot = MX.sym("s1_dot")
    sdot1_dot = MX.sym("sdot1_dot")

    xdot = vertcat(x1_dot, y1_dot, theta1_dot, v1_dot, s1_dot, sdot1_dot)

    # dynamics
    cos_theta = cos(theta1)
    sin_theta = sin(theta1)
    f_expl = vertcat(
        v1 * cos_theta,
        v1 * sin_theta,
        w,
        a,
        sdot1,
        sddot,
    )

    f_impl = xdot - f_expl

    # cost
    Q_c = MX.sym("Q_c")  # 0.1
    Q_l = MX.sym("Q_l")  # 100
    Q_s = MX.sym("Q_s")  # 0.5
    Q_a = 1e-1
    Q_w = 1
    Q_sdd = 1e-1

    cost_expr = (
        Q_c * e_c**2
        + Q_l * e_l**2
        + Q_a * a**2
        + Q_w * w**2
        + Q_sdd * sddot**2
        - Q_s * sdot1
    )

    cost_expr_e = Q_c * e_c**2 + Q_l * e_l**2 - Q_s * sdot1
    # cost_expr = y_expr.T @ Q_mat @ y_expr - Q_s * sdot1
    # cost_expr_e = y_expr_e.T @ Q_mat_e @ y_expr_e - Q_s * sdot1

    # Control Barrier Function
    alpha = MX.sym("alpha")

    # vector pointing towards the obstacle from the robot
    # roughly equal to norm of trajectory derivative since
    # lag error is nearly 0
    obs_dirx = -yr_dot / sqrt(xr_dot**2 + yr_dot**2)
    obs_diry = xr_dot / sqrt(xr_dot**2 + yr_dot**2)

    signed_d = (x1 - xr) * obs_dirx + (y1 - yr) * obs_diry

    # h_abv = d_abv - signed_d
    # h_blw = signed_d - d_blw

    p_abv = obs_dirx * cos_theta + obs_diry * sin_theta + v1 * 0.05
    h_abv = (d_abv - signed_d - 0.2) * exp(-p_abv)

    p_blw = -obs_dirx * cos_theta - obs_diry * sin_theta + v1 * 0.05
    h_blw = (signed_d - d_blw - 0.2) * exp(-p_blw)

    f = vertcat(v1 * cos_theta, v1 * sin_theta, 0, 0, sdot1, 0)
    g = vertcat(
        horzcat(0, 0, 0),
        horzcat(0, 0, 0),
        horzcat(0, 1, 0),
        horzcat(1, 0, 0),
        horzcat(0, 0, 0),
        horzcat(0, 0, 1),
    )

    # HOCBF (second order)
    # dot_h_abv = jacobian(h_abv, x) @ f
    # dot_h_blw = jacobian(h_blw, x) @ f

    # ddot_h_abv = jacobian(dot_h_abv, x) @ f + jacobian(dot_h_abv, x) @ g @ u
    # ddot_h_blw = jacobian(dot_h_blw, x) @ f + jacobian(dot_h_blw, x) @ g @ u

    # ddot{h} >= -\Kappa * \eta, where \eta = {h, \dot{h}}

    # g_u = vertcat(0, 0, w, a, 0, sddot)
    h_dot_abv = jacobian(h_abv, x)
    Lfh_abv = h_dot_abv @ f

    h_dot_blw = jacobian(h_blw, x)
    Lfh_blw = h_dot_blw @ f

    # con_abv = signed_d - d_abv
    # con_blw = d_blw - signed_d
    con_abv = Lfh_abv + h_dot_abv @ g @ u + alpha * h_abv
    con_blw = Lfh_blw + h_dot_blw @ g @ u + alpha * h_blw

    # con_abv = ddot_h_abv + 0.5 * dot_h_abv + alpha * h_abv
    # con_blw = ddot_h_blw + 0.5 * dot_h_blw + alpha * h_blw

    # Control Lyapunov Function
    Ql_c = MX.sym("Ql_c")
    Ql_l = MX.sym("Ql_l")
    gamma = MX.sym("gamma")

    v = Ql_c * e_c**2 + Ql_l * e_l**2  # + Q_s * (sdot1 - v1) ** 2
    v_dot = jacobian(v, x) @ f + jacobian(v, x) @ g @ u

    lyap_con = v_dot + gamma * v

    p = vertcat(
        x_coeff,
        y_coeff,
        d_abv_coeff,
        d_blw_coeff,
        Q_c,
        Q_l,
        Q_s,
        alpha,
        Ql_c,
        Ql_l,
        gamma,
    )

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
    model.cost_expr_ext_cost = cost_expr
    model.cost_expr_ext_cost_e = cost_expr_e
    model.xdot = xdot
    model.name = model_name

    # no setting cbf for con_h_expr_e since no u in final step
    # model.con_h_expr_0 = vertcat(con_abv, con_blw, lyap_con)
    # model.con_h_expr = vertcat(con_abv, con_blw, lyap_con)

    model.con_h_expr_0 = vertcat(lyap_con)
    model.con_h_expr = vertcat(lyap_con)

    # store meta information
    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        r"$\theta$ [rad]",
        "$v$ [m]",
        "$s$ []",
        "$sdot$ []",
    ]
    model.u_labels = ["$a$", "$w$", "$sddot$"]
    model.t_label = "$t$ [s]"

    return model


if __name__ == "__main__":

    # export_mpcc_ode_model_spline_param()
    export_mpcc_ode_model_spline_tube()
