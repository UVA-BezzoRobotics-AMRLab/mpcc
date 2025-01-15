import numpy as np
from acados_template import AcadosModel
from casadi import (
    MX,
    vertcat,
    sin,
    cos,
    atan2,
    jacobian,
    interpolant,
    bspline,
    Function,
    DM,
)


def export_mpcc_ode_model(arc_len, x_ref, y_ref) -> AcadosModel:

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

    # external model parameters
    # arc_len = np.linspace(0, 100, 1000)
    # x_ref = np.cos(arc_len)
    # y_ref = np.sin(arc_len)

    x_spline = interpolant("x_spline", "bspline", [arc_len], x_ref)
    y_spline = interpolant("y_spline", "bspline", [arc_len], y_ref)

    xr = x_spline(s1)
    yr = y_spline(s1)

    xr_dot = jacobian(xr, s1)
    yr_dot = jacobian(yr, s1)

    phi_r = atan2(xr_dot, yr_dot)

    e_c = sin(phi_r) * (x1 - xr) - cos(phi_r) * (y1 - yr)
    e_l = -cos(phi_r) * (x1 - xr) - sin(phi_r) * (y1 - yr)

    Q_c = 50
    Q_l = 5
    Q_mat = np.diag([Q_c, Q_l, 1e-1, 1e-3, 1e-1])
    Q_mat_e = np.diag([Q_c, Q_l]) / 10

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
    model.xdot = xdot
    model.name = model_name

    model.cost_expr_ext_cost = y_expr.T @ Q_mat @ y_expr - sdot1
    model.cost_expr_ext_cost_e = y_expr_e.T @ Q_mat_e @ y_expr_e - sdot1

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
    Q_mat = np.diag([Q_c, Q_l, 1e-1, 4e-1, 1e-1])
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


if __name__ == "__main__":
    # arc_len = np.linspace(0, 100, 1000)
    # x_ref = np.cos(arc_len)
    # y_ref = np.sin(arc_len)

    # x_spline = interpolant("x_spline", "bspline", [arc_len], x_ref)
    # y_spline = interpolant("y_spline", "bspline", [arc_len], y_ref)

    # export_mpcc_ode_model(arc_len, x_ref, y_ref)
    export_mpcc_ode_model_spline_param()
