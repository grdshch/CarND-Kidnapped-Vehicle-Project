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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 5;
    particles.reserve(num_particles);

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_t(theta, std[2]);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned int i = 0; i < num_particles; ++i) {
        particles.emplace_back(Particle(i, dist_x(gen), dist_y(gen), dist_t(gen), 1));
    }
    weights.resize(num_particles, 1);

    is_initialized = true;

	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    for (auto& p: particles) {
        if (fabs(yaw_rate) < 0.0001) {
            p.x += velocity * delta_t * std::cos(p.theta);
            p.y += velocity * delta_t * std::sin(p.theta);
        }
        else {
            p.x += velocity / yaw_rate * (std::sin(p.theta + yaw_rate * delta_t) - std::sin(p.theta));
            p.y += velocity / yaw_rate * (std::cos(p.theta) - std::cos(p.theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> dist_x(0.0, std_pos[0]);
        std::normal_distribution<double> dist_y(0.0, std_pos[1]);
        std::normal_distribution<double> dist_t(0.0, std_pos[2]);
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_t(gen);

        /*

        std::normal_distribution<double> dist_x(p.x, std_pos[0]);
        std::normal_distribution<double> dist_y(p.y, std_pos[1]);
        std::normal_distribution<double> dist_t(p.theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_t(gen);
    */
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto& obs: observations) {
        auto closest = std::min_element(predicted.begin(), predicted.end(),
                                    [&](LandmarkObs l1, LandmarkObs l2) {
                                        double dist1 = (l1.x - obs.x) * (l1.x - obs.x) + (l1.y - obs.y) * (l1.y - obs.y);
                                        double dist2 = (l2.x - obs.x) * (l2.x - obs.x) + (l2.y - obs.y) * (l2.y - obs.y);
                                        return dist1 < dist2;
                                    });
        obs.id = closest->id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

    double C = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
    for (auto& p: particles) {
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;
        std::vector<Map::single_landmark_s> closest_landmarks;
        for (auto& lm: map_landmarks.landmark_list) {
            if (dist(lm.x_f, lm.y_f, p.x, p.y) < sensor_range) {
                closest_landmarks.push_back(lm);
            }
        }

        p.weight = 1.;
        for (auto& obs: observations) {
            LandmarkObs map_obs;
            map_obs.x = obs.x * std::cos(p.theta) - obs.y * std::sin(p.theta) + p.x;
            map_obs.y = obs.x * std::sin(p.theta) + obs.y * std::cos(p.theta) + p.y;

            auto closest_lm_it = std::min_element(closest_landmarks.begin(), closest_landmarks.end(),
                                            [&](Map::single_landmark_s l1, Map::single_landmark_s l2) {
                                                double d1 = dist(l1.x_f, l1.y_f, map_obs.x, map_obs.y);
                                                double d2 = dist(l2.x_f, l2.y_f, map_obs.x, map_obs.y);
                                                return d1 < d2;
                                            });

            associations.push_back(closest_lm_it->id_i);
            sense_x.push_back(map_obs.x);
            sense_y.push_back(map_obs.y);

            double x_part = 0.5f * std::pow(map_obs.x - closest_lm_it->x_f, 2) / pow(std_landmark[0], 2);
            double y_part = 0.5f * std::pow(map_obs.y - closest_lm_it->y_f, 2) / pow(std_landmark[1], 2);
            p.weight *= std::exp(-(x_part + y_part)) / C;
        }
        SetAssociations(p, associations, sense_x, sense_y);
        //dataAssociation(predicted, observations);

    }
    for (unsigned int i = 0; i < num_particles; ++i) {
        //particles[i].weight /= weight_sum;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> resampled;
    resampled.resize(num_particles);

    weights.clear();
    for (auto& p: particles) {
        weights.push_back(p.weight);
    }

    std::discrete_distribution<unsigned int> dist(weights.begin(), weights.end());

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned int i = 0; i < num_particles; ++i) {
        unsigned int sample = dist(gen);
        resampled[i] = particles[sample];
    }
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
