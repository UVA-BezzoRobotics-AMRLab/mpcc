import numpy as np
from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, atan2, jacobian, interpolant


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

    p = vertcat(xr, yr, xr_dot, yr_dot)

    e_c = sin(phi_r) * (x1 - xr) - cos(phi_r) * (y1 - yr)
    e_l = -cos(phi_r) * (x1 - xr) - sin(phi_r) * (y1 - yr)

    Q_c = 8
    Q_l = 5
    Q_mat = np.diag([Q_c, Q_l])

    y_expr = vertcat(e_c, e_l)

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
        s1_dot,
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

    model.cost_expr_ext_cost = y_expr.T @ Q_mat @ y_expr
    model.cost_expr_ext_cost_e = y_expr.T @ Q_mat @ y_expr

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
    arc_len = np.linspace(0, 100, 1000)
    x_ref = np.cos(arc_len)
    y_ref = np.sin(arc_len)

    # x_spline = interpolant("x_spline", "bspline", [arc_len], x_ref)
    # y_spline = interpolant("y_spline", "bspline", [arc_len], y_ref)

    export_mpcc_ode_model(arc_len, x_ref, y_ref)
