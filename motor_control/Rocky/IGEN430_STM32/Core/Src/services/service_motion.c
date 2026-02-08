/*
 * service_motion.c
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#pragma once
#include <math.h>

#include "services/service_motion.h"
#include "robot.h"

// Define motion parameters (single definition)
float ramp_time = MAX_CART_VEL / MAX_CART_ACC;            // time to ramp to max cart vel
float ramp_dist = 0.5f * MAX_CART_ACC * ramp_time;       // distance moved during ramp


// Robot Cartesian Path Planner
// Simple trapezoidal velocity profile

void motion_plan(robot_t *robot) {

	robot->flag_path_done = false;

	float t1, t2, t3, t_est, t_extra;
	float target_arrival_time = robot->current_target.t_arrival;

	vec3 target = robot->current_target.pos;
	vec3 current_pos = robot->current_pos;

	move_plan *plan = &robot->current_move_plan;

	float D = calc_dist(current_pos, target);

	if (D <= 2.0f * ramp_dist) {
		// Triangle profile: never reaches vmax
		t1 = sqrtf(D / MAX_CART_ACC);
		t2 = 0.0f;
		t3 = t1;
	} else {
		// Trapezoid profile
		t1 = ramp_time;
		t2 = (D - 2.0f * ramp_dist) / MAX_CART_VEL;
		t3 = ramp_time;
	}

	t_est = t1+t2+t3;
	t_extra = robot->current_target.t_arrival - (t1+t2+t3);

	if (t_extra <= 0) {
		t_extra = 0;
	}

	plan->t1 = t_extra;
	plan->t2 = plan->t1 + t1;
	plan->t3 = plan->t2 + t2;
	plan->T = plan->t3 + t3;
	plan->active = true;

};

void motion_start(robot_t *robot) {
	robot->flag_ready_to_move = true;
}

// Calculates the cartesian distance between current pos and target
float calc_dist(vec3 current, vec3 target) {

	float x = target.x - current.x;
	float y = target.y - current.y;
	float z = target.z - current.z;

	return sqrtf(x*x + y*y + z*z);
}

// changes target to defined robot home position
void motion_make_home_target(robot_t *robot) {

	robot->current_target.pos = home;
	robot->current_target.t_arrival = 3;		//(s)
}

void motion_plan_strike(robot_t *robot) {

	// Random strike sequence for now; basically have faddle move 50 mm forward in 2 secs
	robot->current_target.pos.x += 50;			// (mm)
	robot->current_target.t_arrival = 2;		// (s)

}

void motion_strike(robot_t *robot) {

	robot->flag_ready_to_move = true;

}

