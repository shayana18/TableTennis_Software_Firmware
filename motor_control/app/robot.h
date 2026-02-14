#pragma once

#include "shared_types.h"

/*
 * robot_params.h
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */




// ROBOT CONSTANTS
#define MAX_JOINT_ANGLE_LIMIT 0.0
#define MIN_JOINT_ANGLE_LIMIT 0.0
#define MAX_JOINT_VEL 0.0
#define MAX_JOINT_ACC 0.0

// TO FILL IN WITH REAL VALUES
#define MAX_CART_VEL 0.0
#define MAX_CART_ACC 0.0

// TO FILL IN WITH REAL VALUES
#define HOME_X = 0.0
#define HOME_Y = 0.0
#define HOME_Z = 0.0

#define BASE_RADIUS 165 			  // (mm)
#define EE_RADIUS 50				    // (mm)
#define UPPER_ARM_LENGTH 350		// (mm)
#define LOWER_ARM_LENGTH 1000		// (mm)

// TO FILL IN WITH REAL VALUES
#define LIMIT_POS_X
#define LIMIT_NEG_X
#define LIMIT_POS_Y
#define LIMIT_NEG_Y
#define LIMIT_POS_Z
#define LIMIT_NEG_Z


vec3 home = {HOME_X, HOME_Y, HOME_Z};


// Target Types
typedef enum {
  TARGET_NONE = 0,
  TARGET_INTERCEPT,
  TARGET_STRIKE,
  TARGET_HOME
} target_type;

// Target Structure
typedef struct {
  target_type type;
  float target_ID;
  vec3 pos;
  float t_arrival; 				// (ms)
  float timestamp;				// (time?)
} target_t;

typedef enum {
  STATE_OFF = 0,
  STATE_UNHOMED,
  STATE_PLAN,
  STATE_MOVE,
  STATE_IDLE,
  STATE_STRIKE,
  STATE_FAULT
} robot_state;



// Move Path Structure
typedef struct {
    vec3 dir;       			// unit direction to target
    float D;        			// total distance
    float t1;       			// acc start time
    float t2;       			// acc end time
    float t3;       			// cruise end time
    float T;        			// total duration
    uint32_t t_start_ms; 		// when plan started
    bool active;
} move_plan;

// ROBOT OBJECT
typedef struct {
  robot_state state;
  vec3 current_pos;

  // current target the FSM is working on
  target_t current_target;
  move_plan current_move_plan;

  // flags set by comms / safety / ISR
  volatile bool flag_new_target;
  volatile bool flag_ready_to_move;
  volatile bool flag_path_done;     // set when execution finishes
  volatile bool flag_path_abort;    // set when ISR hits limit/singularity/etc
  volatile bool flag_fault;         // any fault condition
  volatile bool flag_pc_error;      // optional: PC comm watchdog
} robot_t;



// FUNCTIONS

vec3 robot_get_current_pos();
vec3 FK(float motor_q1, float motor_q2, float motor_q3);
int IK(float x0, float y0, float z0, float *t1, float *t2, float *t3);
void set_idle(robot_t *robot);
void stop_motion(void);
void safety_enter_fault_mode(void);


