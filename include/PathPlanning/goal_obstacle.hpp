#pragma once
#include <string>
#include <cmath>
#include <Eigen/Dense>


namespace PathPlanning{
	struct Goal{
		Eigen::Vector2d position {0.0, 0.0};
		double k_att {1.0};

		inline void setPosition(Eigen::Vector2d& newPosition) noexcept { position = newPosition;}
		inline void setAttractiveGain(double newForce) noexcept {k_att = newForce;}

		inline const double getAttractiveGain() const noexcept {return k_att;}
		inline const Eigen::Vector2d getPosition() const noexcept {return position;}

		inline const double getAttractiveForce(Eigen::Vector2d point) const noexcept{
			return 0.5*k_att * (point-position).squaredNorm();
		}
		inline const Eigen::Vector2d getAttractiveGradient(Eigen::Vector2d point) const noexcept{
			return -k_att * (point - position);
		}

		};
	
	struct Obstacle {

		std::string id;
		Eigen::Vector2d position {0.0, 0.0};
		double amplitude {1.0}; // strength
		double sigma {0.5}; //width
			
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
			return sigma;
		}

		inline const Eigen::Vector2d& getPosition() const noexcept{
			return position;
		}
		
		inline double getPotential(Eigen::Vector2d& point) const noexcept{
			Eigen::Vector2d diffs = point - position;
			double ssd = diffs.squaredNorm();
			double coeff = std::exp(-ssd / (2.0 * sigma * sigma));
			return amplitude * coeff;
		}
		inline Eigen::Vector2d getGradient(Eigen::Vector2d point) const noexcept{
			Eigen::Vector2d diffs = point - position;
			double ssd = diffs.dot(diffs);
			double coeff = std::exp(-ssd / (2.0 * sigma * sigma));
			return amplitude * coeff * diffs / (sigma*sigma);
		}
		
	};
}
