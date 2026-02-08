/*
 * service_motion.h
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#ifndef INC_SERVICES_SERVICE_MOTION_H_
#define INC_SERVICES_SERVICE_MOTION_H_

#include "robot.h"

// Motion parameters for trapezoid vel profile (defined in service_motion.c)
extern float ramp_time;   // time to ramp to max cart vel
extern float ramp_dist;   // distance moved during ramp

// MOTION FUNCTIONS
// Generates trapezoid velocity profile to get to target
void motion_plan(robot_t *robot);
void motion_start(robot_t *robot);
void motion_plan_strike(robot_t *robot);
void motion_strike(robot_t *robot);
void motion_make_home_target(robot_t *robot);

// Utility
float calc_dist(vec3 current, vec3 target);

#endif /* INC_SERVICES_SERVICE_MOTION_H_ */
