#include <distance_map_core/distance_map.h>
#include <distance_map_msgs/DistanceMap.h>
#include <mpcc/types.h>
#include <tf/tf.h>

#include <Eigen/Core>
#include <grid_map_ros/grid_map_ros.hpp>
#include <iostream>

extern "C"
{
#include <cpg_solve.h>
#include <cpg_workspace.h>
}

namespace utils
{
/**********************************************************************
 * Function: raycast_grid
 * Description: Casts a ray from a start point in a direction and
 * stores the distance to the first obstacle in the grid map
 * Parameters:
 * @param start: Eigen::Vector2d
 * @param dir: Eigen::Vector2d
 * @param grid_map: const grid_map::GridMap&
 * @param max_dist: double
 * @param actual_dist: double&
 * Returns:
 * bool - true if successful, false otherwise
 * Notes:
 * Will return false if start or end indices not in map
 **********************************************************************/
inline bool raycast_grid(const Eigen::Vector2d& start, const Eigen::Vector2d& dir,
                         const grid_map::GridMap& grid_map, double max_dist,
                         double& actual_dist)
{
    grid_map::Position end = start + max_dist * dir;

    // get indices in map
    grid_map::Index start_ind;
    if (!grid_map.getIndex(start, start_ind)) return false;

    grid_map::Index end_ind;
    if (!grid_map.getIndex(end, end_ind)) return false;

    Eigen::Vector2d ray_end = end;
    for (grid_map::LineIterator iterator(grid_map, start_ind, end_ind); !iterator.isPastEnd();
         ++iterator)
    {
        if (grid_map.at("layer", *iterator) == 100)
        {
            // I'm pretty sure this isn't possible
            if (!grid_map.getPosition(*iterator, ray_end)) return false;

            break;
        }
    }

    actual_dist = (start - ray_end).norm();
    return true;
}

/**********************************************************************
 * Function: Interp
 * Description: Interpolates a 1D spline through a set of points
 * Parameters:
 * @param pts: const Eigen::RowVectorXd&
 * @param degree: Eigen::DenseIndex
 * @param knot_parameters: const Eigen::RowVectorXd&
 * Returns:
 * Spline1D
 * Notes:
 * This function is a wrapper around the Eigen Spline class, and
 * modified only slightly to allow for the not-a-knot condition
 **********************************************************************/
inline Spline1D Interp(const Eigen::RowVectorXd& pts, Eigen::DenseIndex degree,
                       const Eigen::RowVectorXd& knot_parameters)
{
    using namespace Eigen;

    typedef typename Spline1D::KnotVectorType::Scalar Scalar;
    typedef typename Spline1D::ControlPointVectorType ControlPointVectorType;

    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

    Eigen::RowVectorXd knots;
    knots.resize(knot_parameters.size() + degree + 1);

    // not-a-knot condition setup
    knots.segment(0, degree + 1) = knot_parameters(0) * Eigen::RowVectorXd::Ones(degree + 1);
    knots.segment(degree + 1, knot_parameters.size() - 4) =
        knot_parameters.segment(2, knot_parameters.size() - 4);
    knots.segment(knots.size() - degree - 1, degree + 1) =
        knot_parameters(knot_parameters.size() - 1) * Eigen::RowVectorXd::Ones(degree + 1);

    DenseIndex n = pts.cols();
    MatrixType A = MatrixType::Zero(n, n);
    for (DenseIndex i = 1; i < n - 1; ++i)
    {
        const DenseIndex span = Spline1D::Span(knot_parameters[i], degree, knots);

        // The segment call should somehow be told the spline order at compile
        // time.
        A.row(i).segment(span - degree, degree + 1) =
            Spline1D::BasisFunctions(knot_parameters[i], degree, knots);
    }
    A(0, 0)         = 1.0;
    A(n - 1, n - 1) = 1.0;

    HouseholderQR<MatrixType> qr(A);

    // Here, we are creating a temporary due to an Eigen issue.
    ControlPointVectorType ctrls = qr.solve(MatrixType(pts.transpose())).transpose();

    return Spline1D(knots, ctrls);
}

/**********************************************************************
 * Function: setup_lp
 * Description: Sets up the linear program for the CPG solver
 * Parameters:
 * @param d: int
 * @param N: int
 * @param len_start: double
 * @param traj_arc_len: double
 * @param min_dist: double
 * @param dist_vec: const std::vector<double>&
 * Returns:
 * N/A
 * Notes:
 * This function sets up the linear program for the CPG solver, for
 * more details, see Differentiable Collision-Free Parametric Corridors
 * by J. Arrizabalaga, et al.
 **********************************************************************/
inline void setup_lp(int d, int N, double len_start, double traj_arc_len, double min_dist,
                     const std::vector<double>& dist_vec)
{
    // set arc length domain for LP
    cpg_update_Domain(0, 0);
    cpg_update_Domain(1, traj_arc_len);

    double ds = traj_arc_len / (N - 1);

    for (int i = 0; i < N; ++i)
    {
        double s   = i * ds;
        double s_k = 1;
        for (int j = 0; j < d; ++j)
        {
            // matrices are in column-major order for cpg
            // index for -polynomial <= -min_dist constraint
            size_t ind_upper = i + j * 2 * N;
            // index for polynomial <= obs_dist constraint
            size_t ind_lower = i + N + j * 2 * N;

            cpg_update_A_mat(ind_upper, -s_k);
            cpg_update_A_mat(ind_lower, s_k);
            s_k *= s;
        }

        // index for -polynomial <= -min_dist constraint
        cpg_update_b_vec(i, -min_dist);
        // index for polynomial <= obs_dist constraint
        cpg_update_b_vec(i + N, dist_vec[i]);
    }
}

/**********************************************************************
 * Function: get_tubes
 * Description: Generates the upper and lower tubes for a trajectory
 * Parameters:
 * @param d: int
 * @param N: int
 * @param max_dist: double
 * @param traj: const std::array<Spline1D, 2>&
 * @param traj_arc_len: double
 * @param len_start: double
 * @param horizon: double
 * @param grid_map: const grid_map::GridMap&
 * @param tubes: std::array<Eigen::VectorXd, 2>&
 * Returns:
 * bool - true if successful, false otherwise
 * Notes:
 * This function generates the upper and lower tubes for a trajectory
 * using the CPG solver. For more details, see Differentiable Collision-Free
 * Parametric Corridors by J. Arrizabalaga, et al. Each tube is
 * represented as a polynomial of degree d, parameterized by arc len.
 **********************************************************************/
inline bool get_tubes(int d, int N, double max_dist, const std::array<Spline1D, 2>& traj,
                      double traj_arc_len, double len_start, double horizon,
                      const grid_map::GridMap& grid_map, std::array<Eigen::VectorXd, 2>& tubes)
{
    /*************************************
    ********* Get Traj Distances *********
    **************************************/
    double min_dist_abv = 1e6;
    double min_dist_blw = 1e6;
    // double ds = traj_arc_len / (N-1);
    double ds = horizon / (N - 1);

    std::vector<double> ds_above;
    std::vector<double> ds_below;
    ds_above.resize(N);
    ds_below.resize(N);

    for (int i = 0; i < N; ++i)
    {
        double s  = len_start + i * ds;
        double px = traj[0](s).coeff(0);
        double py = traj[1](s).coeff(0);

        double tx = traj[0].derivatives(s, 1).coeff(1);
        double ty = traj[1].derivatives(s, 1).coeff(1);

        double nx = traj[0].derivatives(s, 2).coeff(2);
        double ny = traj[1].derivatives(s, 2).coeff(2);

        Eigen::Vector2d point(px, py);
        Eigen::Vector2d normal(-ty, tx);
        normal.normalize();

        double den       = tx * tx + ty * ty;
        double curvature = fabs(tx * ny - ty * nx) / (den * sqrt(den));  // normal.norm();

        // raycast in direction of normal to find obs dist
        double dist_above;
        if (!raycast_grid(point, normal, grid_map, max_dist, dist_above)) return false;

        double dist_below;
        if (!raycast_grid(point, -1 * normal, grid_map, max_dist, dist_below)) return false;

        if (curvature > 1e-1 && curvature > 1 / (2 * max_dist))
        {
            Eigen::Vector2d n_vec(nx, ny);
            Eigen::Vector2d abv_n_vec(-ty, tx);

            if (n_vec.dot(abv_n_vec) > 0)
                dist_above = std::min(dist_above, 1 / (2 * curvature));
            else
                dist_below = std::min(dist_above, 1 / (2 * curvature));
        }

        if (dist_above < min_dist_abv) min_dist_abv = dist_above;

        if (dist_below < min_dist_blw) min_dist_blw = dist_below;

        ds_above[i] = dist_above;
        // std::cout << "ds_below is " << dist_below << std::endl;
        ds_below[i] = dist_below;
    }

    /*************************************
    ********** Setup & Solve Up **********
    **************************************/

    // setup_lp(d, N, traj_arc_len, 0, ds_above);
    setup_lp(d, N, len_start, horizon, min_dist_abv / 1.1, ds_above);
    // setup_lp(d, N, horizon, 0, ds_above);
    cpg_solve();

    std::string solved_str = "solved";
    // std::string status = CPG_Info.status;
    // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
    // if (status.find(solved_str) == std::string::npos)
    if (CPG_Info.status != 0)
    {
        std::cout << "LP Above Tube Failed: " << CPG_Info.status << std::endl;
        tubes[0] = Eigen::VectorXd(d);
        tubes[1] = Eigen::VectorXd(d);
        return false;
    }

    for (int i = 0; i < d; ++i) std::cout << CPG_Result.prim->var2[i] << ", ";
    std::cout << std::endl;

    Eigen::VectorXd upper_coeffs;
    upper_coeffs.resize(d);
    for (int i = 0; i < d; ++i) upper_coeffs[i] = CPG_Result.prim->var2[i];

    // cpg_set_solver_default_settings();
    /*************************************
    ********* Setup & Solve Down *********
    **************************************/

    setup_lp(d, N, len_start, horizon, min_dist_blw / 1.1, ds_below);
    // setup_lp(d, N, horizon, 0, ds_below);
    cpg_solve();

    // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
    // if (status.find(solved_str) == std::string::npos)
    if (CPG_Info.status != 0)
    {
        std::cout << "LP Below Tube Failed: " << CPG_Info.status << std::endl;
        tubes[0] = Eigen::VectorXd(d);
        tubes[1] = Eigen::VectorXd(d);
        return false;
    }

    for (int i = 0; i < d; ++i) std::cout << CPG_Result.prim->var2[i] << ", ";
    std::cout << std::endl;

    Eigen::VectorXd lower_coeffs;
    lower_coeffs.resize(d);
    for (int i = 0; i < d; ++i) lower_coeffs[i] = CPG_Result.prim->var2[i];

    tubes[0] = upper_coeffs;
    tubes[1] = -1 * lower_coeffs;

    // ecos_workspace = 0;

    return true;
}

/**********************************************************************
 * Function: eval_traj
 * Description: Evaluates a polynomial at a given point
 * Parameters:
 * @param coeffs: const Eigen::VectorXd&
 * @param x: double
 * Returns:
 * double
 * Notes:
 * This function evaluates a polynomial (trajectory) at a given point
 **********************************************************************/
inline double eval_traj(const Eigen::VectorXd& coeffs, double x)
{
    double ret   = 0;
    double x_pow = 1;

    for (int i = 0; i < coeffs.size(); ++i)
    {
        ret += coeffs[i] * x_pow;
        x_pow *= x;
    }

    return ret;
}

}  // namespace utils
