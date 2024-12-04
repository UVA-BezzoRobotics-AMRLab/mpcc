from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos


def export_unicycle_ode_model() -> AcadosModel:

    model_name = "unicycle_model"

    # set up states & controls
    x1 = SX.sym("x1")
    y1 = SX.sym("y1")
    theta1 = SX.sym("theta1")
    v1 = SX.sym("v1")

    x = vertcat(x1, y1, theta1, v1)

    a = SX.sym("a")
    w = SX.sym("w")

    u = vertcat(a, w)

    # xdot
    x1_dot = SX.sym("x1_dot")
    y1_dot = SX.sym("y1_dot")
    theta1_dot = SX.sym("theta1_dot")
    v1_dot = SX.sym("v1_dot")

    xdot = vertcat(x1_dot, y1_dot, theta1_dot, v1_dot)

    # dynamics
    cos_theta = cos(theta1)
    sin_theta = sin(theta1)
    f_expl = vertcat(
        v1 * cos_theta,
        v1 * sin_theta,
        w,
        a,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        r"$\theta$ [rad]",
        "$v$ [m]",
    ]
    model.u_labels = ["$a$", "$w$"]
    model.t_label = "$t$ [s]"

    return model
