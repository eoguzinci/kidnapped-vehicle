/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// random number generator
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	particles.clear();
	// Set number of particles. This parameter can be tweaked.
	num_particles = 100;

	// Gaussian distribution for sensory noise for x,y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
  	
  // Initialize particles
  for (unsigned int i = 0; i < num_particles; ++i)
  {
  	Particle p;
  	p.id = i;
  	p.weight = 1.0;

  	// Sample x, y, theta from normal distribution
  	p.x = dist_x(gen);
  	p.y = dist_y(gen);
  	p.theta = dist_theta(gen);

  	// append to the particles list
  	particles.push_back(p); 
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	
	for (int i = 0; i < num_particles; ++i){
		Particle& p = particles[i]; // what is th diff between Particle& and Particle

		// Gaussian distribution for sensory noise
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		double x = dist_x(gen);
		double y = dist_y(gen);
		double theta = dist_theta(gen);
	

		double x_pred, y_pred, theta_pred;
		// Motion model
		if(yaw_rate<1e-5){
			x_pred = x + velocity * cos(theta) * delta_t ;
			y_pred = y + velocity * sin(theta) * delta_t ;
			theta_pred = theta;
		} else {
			theta_pred = theta + yaw_rate * delta_t;
			x_pred = x + (velocity/yaw_rate) * (sin(theta_pred)-sin(theta));
			y_pred = y + (velocity/yaw_rate) * (cos(theta) - cos(theta_pred));
		}

		p.x = x_pred;
		p.y = y_pred;
		p.theta = theta_pred;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); ++i){
		LandmarkObs& obs = observations[i];
		
		// initialize the minimum distance as large as possible
		double min_dist = numeric_limits<double>::max();

		// initialize all observation ID to None to see which observation associated with a Landmark
		obs.id = -1;

		for(unsigned int j = 0; j < predicted.size(); j++){
			LandmarkObs& p = predicted[j];

			double distance = dist(obs.x, obs.y, p.x, p.y);

			if(distance<min_dist){
				min_dist = distance;
				obs.id = p.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (unsigned int i = 0; i < particles.size(); i++){
		
		// get the ith particle
		Particle& p = particles[i];

		//transform observations from local coord to global(map) coordinates
		vector<LandmarkObs> transformed_observations; 
		for (unsigned int j = 0; j < observations.size(); j++){
			LandmarkObs obs_tf;
			LandmarkObs obs = observations[j];

			obs_tf.id = obs.id;
			obs_tf.x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
			obs_tf.y = p.y + sin(p.theta)*obs.y - cos(p.theta)*obs.x;

			transformed_observations.push_back(obs_tf);
		}

		// Convert map_landmarks.landmark_list to vector<LandmarkObs>
		vector<LandmarkObs> predicted;
		for (auto& lm_iter : map_landmarks.landmark_list) {
			LandmarkObs lm;

			lm.id = lm_iter.id_i;
			lm.x = lm_iter.x_f;
			lm.y = lm_iter.y_f;

			predicted.push_back(lm);
		}

		// Associating the observations map coordinates of landmarks
		dataAssociation(predicted, transformed_observations);

		// Calculate weight of this particle 
		particles[i].weight = 1.0;

		double std_x = std_landmark[0];
  	double std_y = std_landmark[1];
		for (unsigned int j = 0; j < transformed_observations.size(); j++){
				
				double obs_x, obs_y, pred_x, pred_y;
				obs_x = transformed_observations[j].x;
				obs_y = transformed_observations[j].y;
				int association_id = transformed_observations[j].id;

				// search for associated landmark with the observation
				for(unsigned int k = 0; k < predicted.size(); k++){
					if (predicted[k].id == association_id)
					{
						pred_x = predicted[k].x;
						pred_y = predicted[k].y;
					}
				}

				//multivariate Gaussian
      	double obs_weight = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2))) ) );

      	// multiply the observation weights
      	particles[i].weight *= obs_weight;
		}		

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> new_particles;

	// access current weights
	vector<double> weights;
	for (unsigned int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}

	// Sample from discrete distribution
	discrete_distribution<int> discr_distribution(weights.begin(), weights.end());
	for (unsigned int i=0; i < particles.size(); i++) {
		int rnd_ind = discr_distribution(gen);
		new_particles.push_back(particles[rnd_ind]);
	}

	// update particles
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
