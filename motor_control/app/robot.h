#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "shared_types.h"

// Robot motion limits
#define MAX_JOINT_ANGLE_LIMIT 180.0f
#define MIN_JOINT_ANGLE_LIMIT -180.0f
#define MAX_JOINT_VEL 720.0f
#define MAX_JOINT_ACC 3600.0f

#define MAX_CART_VEL 400.0f     // mm/s
#define MAX_CART_ACC 2000.0f    // mm/s^2

// Home position (mm)
#define HOME_X 0.0f
#define HOME_Y 0.0f
#define HOME_Z -700.0f

// Delta geometry (mm)
#define BASE_RADIUS 165.0f
#define EE_RADIUS 50.0f
#define UPPER_ARM_LENGTH 350.0f
#define LOWER_ARM_LENGTH 1000.0f

// Workspace bounds (mm)
#define LIMIT_POS_X 250.0f
#define LIMIT_NEG_X -250.0f
#define LIMIT_POS_Y 250.0f
#define LIMIT_NEG_Y -250.0f
#define LIMIT_POS_Z -400.0f
#define LIMIT_NEG_Z -950.0f

// Motion execution configuration
#define MOTION_EXECUTE_PERIOD_MS 20U
#define JOINT_SPEED_CMD_SCALE 1.0f
#define JOINT_GEAR_RATIO 1.0f
#define MAX_MOTOR_SPEED_CMD 3000L

// Motor IDs on the daisy-chain
#define ROBOT_MOTOR_1_ID 2
#define ROBOT_MOTOR_2_ID 3
#define ROBOT_MOTOR_3_ID 4

extern const vec3 home;

typedef struct {
  target_type type;
  float target_ID;
  vec3 pos;
  float t_arrival_s;
  float timestamp;
} robot_target_t;

typedef enum {
  STATE_OFF = 0,
  STATE_UNHOMED,
  STATE_PLAN,
  STATE_MOVE,
  STATE_IDLE,
  STATE_STRIKE,
  STATE_FAULT
} robot_state;

// Move profile and execution state
typedef struct {
  vec3 start_pos;
  vec3 target_pos;
  vec3 dir;

  float D;
  float t1;
  float t2;
  float t3;
  float T;

  uint32_t t_start_ms;
  uint32_t prev_tick_ms;
  float prev_joint_deg[3];
  bool prev_joint_valid;
  bool active;
} move_plan;

typedef struct {
  robot_state state;
  vec3 current_pos;

  robot_target_t current_target;
  move_plan current_move_plan;

  volatile bool flag_new_target;
  volatile bool flag_ready_to_move;
  volatile bool flag_path_done;
  volatile bool flag_path_abort;
  volatile bool flag_fault;
  volatile bool flag_pc_error;
} robot_t;

// Kinematics
vec3 robot_get_current_pos(void);
vec3 FK(float motor_q1, float motor_q2, float motor_q3);
int IK(float x0, float y0, float z0, float *t1, float *t2, float *t3);

// Robot helpers
void robot_set_target_from_mail(robot_target_t *dst, const target_t *src);
float robot_calc_dist(vec3 current, vec3 target);
bool robot_target_in_workspace(vec3 pos);

// Safety/state helpers
void set_idle(robot_t *robot);
void stop_motion(void);
void safety_enter_fault_mode(void);
