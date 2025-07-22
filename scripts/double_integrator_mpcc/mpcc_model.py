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

def create_mpcc_ode_model() -> AcadosModel:

    model_name = "double_integrator_model"

    x1 = MX.sym("x1")
    y1 = MX.sym("y1")
    vx1 = MX.sym("vx1")
    vy1 = MX.sym("vy1")
    s1 = MX.sym("s1")
    sdot1 = MX.sym("s1_dot")

    x = vertcat(x1, y1, vx1, vy1, s1, sdot1)

    ax = MX.sym("ax")
    ay = MX.sym("ay")
    sddot = MX.sym("sddot")

    u = vertcat(ax, ay, sddot)

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

    y_expr = vertcat(e_c, e_l, ax, ay, sddot)
    y_expr_e = vertcat(e_c, e_l)

    # xdot
    x1_dot = MX.sym("x1_dot")
    y1_dot = MX.sym("y1_dot")
    vx1_dot = MX.sym("vx1_dot")
    vy1_dot = MX.sym("vy1_dot")
    s1_dot = MX.sym("s1_dot")
    sdot1_dot = MX.sym("sdot1_dot")

    x_dot = vertcat(x1_dot, y1_dot, vx1_dot, vy1_dot, s1_dot, sdot1_dot)

    # dynamics
    f_expl = vertcat(
        vx1,
        vy1,
        ax,
        ay,
        sdot1,
        sddot,
    )

    f_impl = x_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
    model.xdot = x_dot
    model.name = model_name

    model.cost_expr_ext_cost = y_expr.T @ Q_mat @ y_expr - 0.2 * sdot1
    model.cost_expr_ext_cost_e = y_expr_e.T @ Q_mat_e @ y_expr_e - 0.2 * sdot1

    # store meta information
    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        "$v_x$ [m/s]",
        "$v_y$ [m/s]",
        "$s$ []",
        "$sdot$ []",
    ]
    model.u_labels = ["$ax$", "$ay$", "$sddot$"]
    model.t_label = "$t$ [s]"

    return model
