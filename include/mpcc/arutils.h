#pragma once

#include <vector>
#include <Eigen/Core>
#include <uvatraj_msgs/ControlPoint.h>
#include <string>
namespace PathPlanning{
	struct Goal{
		Eigen::Vector2d position;
		double k_att;


		inline void setPosition(Eigen::Vector2d& newPosition) { position = newPosition}
		inline void setAttractiveGain(double newForce) {k_att = newForce}

		inline const double getAttractiveGain() {return k_att}
		inline const Eigen::Vector2d getPosition() {return position}

		inline const Eigen::Vector2d getAttractiveForce(Eigen::Vector2d point){
			return 0.5*k_att * (point-position).squared.Norm()
		}
		inline const Eigen::Vector2d getAttractiveGradient(Eigen::Vector2d point){
			return -k_att * (point * goal);
		}

		};
	
	struct Obstacle {
		std::string id;
		Eigen::Vector2d position;
		double amplitude; // strength
		double sigma; //width
			

		//Setters
		
		inline void setPosition(Eigen::Vector2d& newPosition){
			position = newPosition;
		}

		inline void setParams(double newAmplitude, double newSigma){
			amplitude = newAmplitude;
			sigma = newSigma;
		}
		
		//Getters

		inline const double getAmplitude() const noexcept{
			return amplitude;
		}

		inline const double getSigma() const noexcept{
			return sigma
		}

		inline const Eigen::Vector2d& getPosition() const noexcept{
			return position
		}
		
		inline double getPotential(Eigen::Vector2d& point){
			Eigen::Vector2d diffs = point - position
			double ssd = diff.squaredNorm();
			double coeff = std::exp(-ssd / (2.0 * sigma * sigma))
			return amplitude * coeff
		}
		inline Eigen::Vector2d getGradient(Eigen::Vector2d point){
			Eigen::Vector2d diffs = point - position
			double ssd = diffs.dot(diffs)
			double coeff = std::exp(-ssd / (2.0 * sigma * sigma))
			return amplitude * coeff * diffs / (sigma*sigma)
		}
		
	};

	namespace Goal{
		Eigen::Vector2d getGoalGradient(Eigen::Vector2d point);
		}

	namespace GaussianPotentialField{
		Eigen::Vector2d getTotalGradient(Eigen::Vector2d point);
		std::vector<Eigen::Vector2d> generateTrajectory(Eigen::Vector2d pos_g);
	}
}  
