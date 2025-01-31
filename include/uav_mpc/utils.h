#include <tf/tf.h>
#include <iostream>  
#include <Eigen/Core>

#include <uav_mpc/types.h>

#include <grid_map_ros/grid_map_ros.hpp>

#include <distance_map_msgs/DistanceMap.h>
#include <distance_map_core/distance_map.h>

extern "C"
{
    #include <cpg_solve.h>
    #include <cpg_workspace.h>
}


namespace utils{
    inline distmap::DistanceMap distmap_from_msg(const distance_map_msgs::DistanceMap &msg)
    {
        distmap::DistanceMap grid(distmap::DistanceMap::Dimension(msg.info.width, msg.info.height),
                                msg.info.resolution,
                                distmap::DistanceMap::Origin(msg.info.origin.position.x,
                                                            msg.info.origin.position.y,
                                                            tf::getYaw(msg.info.origin.orientation)));

        std::copy(msg.data.data(),
                msg.data.data() + (msg.info.width * msg.info.height),
                grid.data());

        return grid;
    }

    inline bool raycast_grid(const Eigen::Vector2d& start, 
                             const Eigen::Vector2d& dir, 
                             const grid_map::GridMap& grid_map,
                             double max_dist,
                             double& actual_dist)
    {
        grid_map::Position end = start + max_dist * dir;

        // get indices in map
        grid_map::Index start_ind;
        if (!grid_map.getIndex(start, start_ind))
            return false;

        grid_map::Index end_ind;
        if (!grid_map.getIndex(end, end_ind))
            return false;

        Eigen::Vector2d ray_end = end;
        for(grid_map::LineIterator iterator(grid_map, start_ind, end_ind);
            !iterator.isPastEnd(); ++iterator)
        {
            if(grid_map.at("layer", *iterator) == 100)
            {

                // I'm pretty sure this isn't possible
                if (!grid_map.getPosition(*iterator, ray_end))
                    return false;

                break;
            }
        }

        actual_dist = (start - ray_end).norm();
        return true;
    }

    inline Spline1D Interp(const Eigen::RowVectorXd &pts, Eigen::DenseIndex degree, const Eigen::RowVectorXd &knot_parameters)
    {
        using namespace Eigen;
        
        typedef typename Spline1D::KnotVectorType::Scalar Scalar;
        typedef typename Spline1D::ControlPointVectorType ControlPointVectorType;

        typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

        Eigen::RowVectorXd knots;
        knots.resize(knot_parameters.size() + degree + 1);

        // not-a-knot
        knots.segment(0, degree + 1) = knot_parameters(0) * Eigen::RowVectorXd::Ones(degree + 1);
        knots.segment(degree + 1, knot_parameters.size() - 4) = knot_parameters.segment(2, knot_parameters.size() - 4);
        knots.segment(knots.size() - degree - 1, degree + 1) = knot_parameters(knot_parameters.size() - 1) * Eigen::RowVectorXd::Ones(degree + 1);

        DenseIndex n = pts.cols();
        MatrixType A = MatrixType::Zero(n, n);
        for (DenseIndex i = 1; i < n - 1; ++i)
        {
            const DenseIndex span = Spline1D::Span(knot_parameters[i], degree, knots);

            // The segment call should somehow be told the spline order at compile time.
            A.row(i).segment(span - degree, degree + 1) = Spline1D::BasisFunctions(knot_parameters[i], degree, knots);
        }
        A(0, 0) = 1.0;
        A(n - 1, n - 1) = 1.0;

        HouseholderQR<MatrixType> qr(A);

        // Here, we are creating a temporary due to an Eigen issue.
        ControlPointVectorType ctrls = qr.solve(MatrixType(pts.transpose())).transpose();

        return Spline1D(knots, ctrls);
    }


    inline void setup_lp(int d, 
                         int N, 
                         double traj_arc_len, 
                         double min_dist, 
                         const std::vector<double>& dist_vec)
    {

        // set arc length domain for LP
        cpg_update_Domain(0, 0.);
        cpg_update_Domain(1, traj_arc_len);

        double ds = traj_arc_len / (N-1);

        for(int i = 0; i < N; ++i)
        {
            double s = i * ds;
            double s_k = 1;
            for(int j = 0; j < d; ++j)
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

    inline double eval_traj(const Eigen::VectorXd& coeffs, double x)
    {
        double ret = 0;
        double x_pow = 1;

        for(int i = 0; i < coeffs.size(); ++i)
        {
            ret += coeffs[i] * x_pow;
            x_pow *= x;
        }

        return ret;
    }

    inline bool get_tubes(int d,
                          int N,
                          double max_dist,
                          const std::vector<Spline1D> &traj, 
                          double traj_arc_len,
                          double len_start,
                          double horizon,
                          const grid_map::GridMap& grid_map,
                          std::vector<Eigen::VectorXd>& tubes)
    {
        tubes.clear();

        /*************************************
        ********* Get Traj Distances *********
        **************************************/
        double min_dist_abv = 1e6;
        double min_dist_blw = 1e6;
        // double ds = traj_arc_len / (N-1);
        double ds = horizon / (N-1);

        std::vector<double> ds_above;
        std::vector<double> ds_below;
        ds_above.resize(N);
        ds_below.resize(N);

        for(int i = 0; i < N; ++i)
        {
            double s = len_start + i * ds;
            double px = traj[0](s).coeff(0);
            double py = traj[1](s).coeff(0);

            double tx = traj[0].derivatives(s, 1).coeff(1);
            double ty = traj[1].derivatives(s, 1).coeff(1);

            Eigen::Vector2d point(px, py);
            Eigen::Vector2d normal(-ty, tx);
            normal.normalize();

            // raycast in direction of normal to find obs dist
            double dist_above;
            if (!raycast_grid(point, normal, grid_map, max_dist, dist_above))
                return false;

            double dist_below;
            if (!raycast_grid(point, -1*normal, grid_map, max_dist, dist_below))
                return false;

            if (dist_above < min_dist_abv)
                min_dist_abv = dist_above;

            if (dist_below < min_dist_blw)
                min_dist_blw = dist_below;

            ds_above[i] = dist_above;
            // std::cout << "ds_below is " << dist_below << std::endl;
            ds_below[i] = dist_below;
        }

        /*************************************
        ********** Setup & Solve Up **********
        **************************************/

        // setup_lp(d, N, traj_arc_len, 0, ds_above);
        setup_lp(d, N, horizon, min_dist_abv/1.1, ds_above);
        cpg_solve();

        std::string solved_str = "solved";
        // std::string status = CPG_Info.status;
        // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
        // if (status.find(solved_str) == std::string::npos)
        if (CPG_Info.status != 0)
        {
            std::cout << "LP Above Tube Failed: " << CPG_Info.status << std::endl;
            tubes.push_back(Eigen::VectorXd(d));
            tubes.push_back(Eigen::VectorXd(d));
            return false;
        }

        for(int i = 0; i < d; ++i)
            std::cout << CPG_Result.prim->var2[i] << ", ";
        std::cout << std::endl;

        Eigen::VectorXd upper_coeffs;
        upper_coeffs.resize(d);
        for(int i = 0; i < d; ++i)
            upper_coeffs[i] = CPG_Result.prim->var2[i];


        // cpg_set_solver_default_settings();
        /*************************************
        ********* Setup & Solve Down *********
        **************************************/

        setup_lp(d, N, horizon, min_dist_blw/1.1, ds_below);
        cpg_solve();

        // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
        // if (status.find(solved_str) == std::string::npos)
        if (CPG_Info.status != 0)
        {
            std::cout << "LP Below Tube Failed: " << CPG_Info.status << std::endl;
            tubes.push_back(Eigen::VectorXd(d));
            tubes.push_back(Eigen::VectorXd(d));
            return false;
        }

        for(int i = 0; i < d; ++i)
            std::cout << CPG_Result.prim->var2[i] << ", ";
        std::cout << std::endl;

        Eigen::VectorXd lower_coeffs;
        lower_coeffs.resize(d);
        for(int i = 0; i < d; ++i)
            lower_coeffs[i] = CPG_Result.prim->var2[i];

        // std::cout << "Result\n" << CPG_Result.info->obj_val << std::endl;
        // std::cout << "Primal Solution\n";
        // cpg_set_solver_default_settings();

        
        // std::cout << std::endl;
        // Eigen::RowVectorXd ss, ds_abv, ds_blw;
        // ss.resize(10);
        // ds_abv.resize(10);
        // ds_blw.resize(10);
        // ds = traj_arc_len / 10.;

        // for(int i = 0; i < 11; ++i)
        // {
        //     double s = i * ds;

        //     ss(i) = s;
        //     ds_abv(i) = eval_traj(upper_coeffs, s);
        //     ds_blw(i) = eval_traj(lower_coeffs, s);
        // }

        // fit splines
        // const auto above_fit = utils::Interp(ds_abv, 3, ss);
        // Spline1D spline_above(above_fit);

        // const auto below_fit = utils::Interp(-1*ds_blw, 3, ss);
        // Spline1D spline_below(below_fit);


        // tubes.push_back(spline_above);
        // tubes.push_back(spline_below);
        tubes.push_back(upper_coeffs);
        tubes.push_back(-1*lower_coeffs);

        // ecos_workspace = 0;

        return true;
    }

}
