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

		inline const double getAttractiveForce(Eigen::Vector2d& point) const noexcept{
			return 0.5*k_att * (point-position).squaredNorm();
		}
		inline const Eigen::Vector2d getAttractiveGradient(Eigen::Vector2d point) const noexcept{
			return k_att * (point - position);
		}

		};
	
	struct Obstacle {

		std::string id;
		Eigen::Vector2d center {0.0, 0.0};
		double amplitude {1.0}; // strength
		double sigma {0.5}; //width
		double radius {5};
		//Setters
	
		inline double dist(const Eigen::Vector2d& p) const noexcept {
        		return (p - center).norm() - radius;
    		}

		inline Eigen::Vector2d normal(const Eigen::Vector2d& p) const noexcept {
			Eigen::Vector2d n = (p - center);
			double nrm = n.norm();
			return n.squaredNorm() > 1e-18 ? n.normalized() : Eigen::Vector2d::Zero();
		}

		inline void setPosition(Eigen::Vector2d& newPosition){
			center = newPosition;
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
			return center;
		}
		
		inline double getPotential(Eigen::Vector2d& point) const noexcept{
			//Eigen::Vector2d diffs = point - position;
			//double ssd = diffs.squaredNorm();
			//double coeff = std::exp(-ssd / (2.0 * sigma * sigma));
			//return amplitude * coeff;
			//	
			double d = std::max(1e-6, dist(point));
        		return amplitude * std::exp(-0.5 * d * d / (sigma * sigma));
		}
		inline Eigen::Vector2d getGradient(Eigen::Vector2d point) const noexcept{
		//	Eigen::Vector2d diffs = point - position;
		//	double ssd = diffs.dot(diffs);
		//	double coeff = std::exp(-ssd / (2.0 * sigma * sigma));
		//	return amplitude * coeff * diffs / (sigma*sigma);

			double d = std::max(1e-6, dist(point));
        		double coeff = amplitude * std::exp(-0.5 * d * d / (sigma * sigma))* d / (sigma * sigma);
        		return coeff * normal(point);                   // points outward

		}
		
	};
}
