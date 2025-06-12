#pragma once

#include <vector>
#include <Eigen/Core>
#include <uvatraj_msgs/ControlPoint.h>

namespace PathPlanning{
	namespace Obstacle {
		struct Obstacle {
			std::string identifier;
			double position;
			double amplitude; // strength
			double sigma; //width
			}
		void setPosition(Eigen::Vector2d position);
		void setUpdateParams(double amplitude, double sigma);
		double getPotential(Eigen::Vector2d point);
		Eigen::Vector2d getGradient(Eigen::Vector2d point);
			
	}
	namespace Goal{
		struct Goal{
			double position;
			double k_att;
			}
		Eigen::Vector2d getGoalGradient(Eigen::Vector2d point);
		}

	namespace GaussianPotentialField{
		Eigen::Vector2d getTotalGradient(Eigen::Vector2d point);
		std::vector<Eigen::Vector2d> generateTrajectory(Eigen::Vector2d pos_g);
	}
}  
