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
    num_particles = 10;
    particles.reserve(num_particles);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    // initialize all particles using GPS coordinates and adding Gaussian noise
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_t(0.0, std_pos[2]);

    // predict next position of every particle
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
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_t(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    double C = std::pow(2.0 * M_PI * std_landmark[0] * std_landmark[1], observations.size());
    for (auto& p: particles) {

        // find all map landmarks within sensor_range of current particle
        std::vector<Map::single_landmark_s> closest_landmarks;
        for (auto& lm: map_landmarks.landmark_list) {
            if (dist(lm.x_f, lm.y_f, p.x, p.y) < sensor_range) {
                closest_landmarks.push_back(lm);
            }
        }

        p.weight = 1.;
        for (auto& obs: observations) {
            // transform observation position from vehicle's to global coordinates
            LandmarkObs map_obs;
            map_obs.x = obs.x * std::cos(p.theta) - obs.y * std::sin(p.theta) + p.x;
            map_obs.y = obs.x * std::sin(p.theta) + obs.y * std::cos(p.theta) + p.y;

            // for each observation find closest map landmark
            double min_dist = 1000;
            Map::single_landmark_s closest_lm;
            for (auto& lm: closest_landmarks) {
                double d = dist(map_obs.x, map_obs.y, lm.x_f, lm.y_f);
                if (d < min_dist) {
                    closest_lm = lm;
                    min_dist = d;
                }
            }
            // calculate multivariate Gaussian probability
            double x_part = 0.5f * std::pow(map_obs.x - closest_lm.x_f, 2) / pow(std_landmark[0], 2);
            double y_part = 0.5f * std::pow(map_obs.y - closest_lm.y_f, 2) / pow(std_landmark[1], 2);
            p.weight *= std::exp(-(x_part + y_part));
        }
        p.weight /= C;
    }
}

void ParticleFilter::resample() {
    std::vector<Particle> resampled;
    resampled.reserve(num_particles);

    for (unsigned int i = 0; i < num_particles; ++i) {
        weights[i] = particles[i].weight;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<unsigned int> dist(weights.begin(), weights.end());

    for (unsigned int i = 0; i < num_particles; ++i) {
        unsigned int sample = dist(gen);
        resampled.push_back(particles[sample]);
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
