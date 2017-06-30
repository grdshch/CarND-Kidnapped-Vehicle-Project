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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
    }
    weights.resize(num_particles, 1);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
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
        std::normal_distribution<double> dist_x(p.x, std_pos[0]);
        std::normal_distribution<double> dist_y(p.y, std_pos[1]);
        std::normal_distribution<double> dist_t(p.theta, std_pos[2]);
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_t(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
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
            double x_part = 0.5f * std::pow(map_obs.x - closest_lm_it->x_f, 2) / pow(std_landmark[0], 2);
            double y_part = 0.5f * std::pow(map_obs.y - closest_lm_it->y_f, 2) / pow(std_landmark[1], 2);
            p.weight *= std::exp(-(x_part + y_part)) / C;
        }
    }
}

void ParticleFilter::resample() {
    std::vector<Particle> resampled;
    resampled.resize(num_particles);

    for (unsigned int i = 0; i < num_particles; ++i) {
        weights[i] = particles[i].weight;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<unsigned int> dist(weights.begin(), weights.end());

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
