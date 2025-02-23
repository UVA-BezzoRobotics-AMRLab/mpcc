#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/math.h"
// #include "acados/utils/print.h"
// #include "acados_c/external_function_interface.h"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/Splines>

#include "acados_c/ocp_nlp_interface.h"
#include "mpcc/unicycle_mpcc/acados_solver_unicycle_model.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#define NX UNICYCLE_MODEL_NX
#define NP UNICYCLE_MODEL_NP
#define NU UNICYCLE_MODEL_NU
#define NBX0 UNICYCLE_MODEL_NBX0

void interpolate_and_compute_derivatives(const Eigen::MatrixXd &reference)
{
    int n = reference.rows();
    Eigen::ArrayXd time_steps =
        Eigen::ArrayXd::LinSpaced(n, 0, 1);  // Normalize time steps to [0, 1]

    // Create splines for x_ref and y_ref
    Eigen::Spline<double, 1> spline_x =
        Eigen::SplineFitting<Eigen::Spline<double, 1> >::Interpolate(
            reference.col(0).transpose(), 3, time_steps);
    Eigen::Spline<double, 1> spline_y =
        Eigen::SplineFitting<Eigen::Spline<double, 1> >::Interpolate(
            reference.col(1).transpose(), 3, time_steps);

    // Evaluate the splines and their derivatives at a specific point
    double t               = 0.5;  // Example point in [0, 1]
    Eigen::VectorXd x_val  = spline_x(t);
    Eigen::VectorXd y_val  = spline_y(t);
    Eigen::VectorXd x_der1 = spline_x.derivatives(t, 1).row(1);
    Eigen::VectorXd y_der1 = spline_y.derivatives(t, 1).row(1);
    Eigen::VectorXd x_der2 = spline_x.derivatives(t, 2).row(2);
    Eigen::VectorXd y_der2 = spline_y.derivatives(t, 2).row(2);

    // Print the results
    std::cout << "x(t): " << x_val << std::endl;
    std::cout << "y(t): " << y_val << std::endl;
    std::cout << "x'(t): " << x_der1 << std::endl;
    std::cout << "y'(t): " << y_der1 << std::endl;
    std::cout << "x''(t): " << x_der2 << std::endl;
    std::cout << "y''(t): " << y_der2 << std::endl;
}

Eigen::MatrixXd get_reference()
{
    double R = 3.;
    double T = 20.;
    int n    = 30;

    Eigen::ArrayXd time_steps = Eigen::ArrayXd::LinSpaced(0, T, n);
    Eigen::ArrayXd thetas     = -2 * M_PI * time_steps / T + M_PI / 2;

    Eigen::ArrayXd x_ref = R * Eigen::cos(thetas);
    Eigen::ArrayXd y_ref = R * Eigen::sin(thetas) - R;

    Eigen::MatrixXd reference(n, 2);
    reference.col(0) = x_ref;
    reference.col(1) = y_ref;

    interpolate_and_compute_derivatives(reference);

    return reference;
}

int main()
{
    unicycle_model_solver_capsule *acados_ocp_capsule = unicycle_model_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C
    // without new code generation
    int N = UNICYCLE_MODEL_N;
    // allocate the array and fill it accordingly
    double *new_time_steps = NULL;
    int status =
        unicycle_model_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("unicycle_model_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    get_reference();
    exit(0);

    ocp_nlp_config *nlp_config = unicycle_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims     = unicycle_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in         = unicycle_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out       = unicycle_model_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = unicycle_model_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts             = unicycle_model_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 0;
    ubx0[1] = 0;
    lbx0[2] = 0;
    ubx0[2] = 0;
    lbx0[3] = 0;
    ubx0[3] = 0;
    lbx0[4] = 0;
    ubx0[4] = 0;
    lbx0[5] = 0;
    ubx0[5] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;
    x_init[4] = 0.0;
    x_init[5] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;

    // prepare evaluation
    int NTIMINGS    = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N + 1)];
    double utraj[NU * N];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        status = unicycle_model_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii * NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii * NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat(NX, N + 1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat(NU, N, utraj, NU);
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("unicycle_model_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("unicycle_model_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    unicycle_model_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n", sqp_iter,
           NTIMINGS, min_time * 1000, kkt_norm_inf);

    // free solver
    status = unicycle_model_acados_free(acados_ocp_capsule);
    if (status)
    {
        printf("unicycle_model_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = unicycle_model_acados_free_capsule(acados_ocp_capsule);
    if (status)
    {
        printf("unicycle_model_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
