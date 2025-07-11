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

    x = vertcat(x1, y1, vx1, vy1)

    ax = MX.sym("ax")
    ay = MX.sym("ay")

    u = vertcat(ax, ay)

    model = AcadosModel()

    return model
