/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <limits>

#include "helper_functions.h"

#define NUM_PARTICLES 500

#define EPS 1e-8
#define DOUBLE_INF std::numeric_limits<double>::infinity()

using std::string;
using std::vector;
using std::array;

using std::normal_distribution;
using std::discrete_distribution;
using std::initializer_list;

using std::cos;
using std::sin;
using std::pow;
using std::sqrt;
using std::exp;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Set number of particles
  num_particles = NUM_PARTICLES; 

  // Define probability distribution for the initial state
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Draw particles from the distribution
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    
    particle.x = dist_x(generator);
    particle.y = dist_x(generator);
    particle.theta = dist_theta(generator);
      
    particles.push_back(particle);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
  // Define probability distribution for random noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  normal_distribution<double> dist_velocity(velocity, 0.1);
  normal_distribution<double> dist_yaw_rate(yaw_rate, 0.1);
  
  // Update particles
  for (int i = 0; i < num_particles; i++) {
    Particle& particle = particles[i];
    
    // Avoid division by zero
    if (fabs(yaw_rate) < EPS) {
      double distance = velocity * delta_t;
      
      particle.x += distance * cos(particle.theta);
      particle.y += distance * sin(particle.theta);
    } else {
      double coeff = velocity / yaw_rate;
      double new_theta = particle.theta + yaw_rate * delta_t;
      
      particle.x += coeff * (sin(new_theta) - sin(particle.theta));
      particle.y += coeff * (cos(particle.theta) - cos(new_theta));
      particle.theta = new_theta;
    }
    
    // Apply random noise
    particle.x += dist_x(generator);
    particle.y += dist_y(generator);
    particle.theta += dist_theta(generator);
  }
}

void ParticleFilter::dataAssociation(const Map& map_landmarks, 
                                     vector<LandmarkObs>& observations) {
  
  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs& observation = observations[i];
    
    // Associate landmark via nearest neighbor
    double nearest_dist = DOUBLE_INF;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      const Map::single_landmark_s& landmark = map_landmarks.landmark_list[j];
      
      double dist = sqrt(pow(landmark.x_f - observation.x, 2)
                       + pow(landmark.y_f - observation.y, 2));
      
      if (dist < nearest_dist) {
        observation.id = j;
        nearest_dist = dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  
  for (int i = 0; i < num_particles; i++) {
    Particle& particle = particles[i];
    vector<LandmarkObs> transformed_observations;
    
    // Transform observations to map coordinate space
    for (unsigned int j = 0; j < observations.size(); j++) {
      const LandmarkObs& observation = observations[j];
      LandmarkObs transformed_observation;
      
      transformed_observation.x = particle.x + cos(particle.theta) * observation.x
                                             - sin(particle.theta) * observation.y;
      
      transformed_observation.y = particle.y + sin(particle.theta) * observation.x
                                             + cos(particle.theta) * observation.y;
      
      transformed_observations.push_back(transformed_observation);
    }
    
    // Associate landmarks to observations
    dataAssociation(map_landmarks, transformed_observations);
    
    // Calculate particle weight
    particle.weight = 1.0;
    for (unsigned int j = 0; j < transformed_observations.size(); j++) {
      const LandmarkObs& observation = transformed_observations[j];
      const Map::single_landmark_s& landmark = map_landmarks.landmark_list[observation.id];
      
      double diff_x = observation.x - landmark.x_f;
      double diff_y = observation.y - landmark.y_f;
      
      double exponent = -(pow(diff_x / std_x, 2) + pow(diff_y / std_y, 2));
      double coeff = 1 / (2 * M_PI * std_x * std_y);
      
      particle.weight *= coeff * exp(exponent);
    }
  }
}

void ParticleFilter::resample() {
  
  // Put all particle weights into a single vector
  array<double, NUM_PARTICLES> weights;
  for (int i = 0; i < num_particles; i++) {
    weights[i] = particles[i].weight;
  }
  
  // Create a discrete distribution from the weight array
  discrete_distribution<int> dist(weights.begin(), weights.end());
  
  // Sample particles from the distribution
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; i++) {
    int idx = dist(generator);
    resampled_particles.push_back(particles[idx]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}