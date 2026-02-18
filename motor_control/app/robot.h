#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "shared_types.h"

// Robot motion limits
#define MAX_JOINT_ANGLE_LIMIT HOME_ANGLE_ABOVE_HOR + 5.0f // Temporary to accound for encoder inaccuracy.
#define MIN_JOINT_ANGLE_LIMIT 70.0f
#define MAX_JOINT_VEL 3000.0L      // RPM, Conservative values
#define MAX_JOINT_ACC 1000.0L     // RPM/s, Conservative Values

#define MAX_CART_VEL 4000.0f     // mm/s
#define MAX_CART_ACC 20000.0f    // mm/s^2

// Home position (mm)
#define HOME_X 0.0f
#define HOME_Y 0.0f
#define HOME_Z -1000.0f

// Delta geometry (mm)
#define BASE_RADIUS 165.0f
#define EE_RADIUS 50.0f
#define UPPER_ARM_LENGTH 350.0f
#define LOWER_ARM_LENGTH 1000.0f

// Workspace bounds (mm)
#define LIMIT_POS_X 500.0f
#define LIMIT_NEG_X -500.0f
#define LIMIT_POS_Y 300.0f
#define LIMIT_NEG_Y -300.0f
#define LIMIT_POS_Z -700.0f
#define LIMIT_NEG_Z -1000.0f

// Motion execution configuration
#define MOTION_EXECUTE_PERIOD_MS 20U
#define JOINT_SPEED_CMD_SCALE 1.0f
#define JOINT_GEAR_RATIO 10.0f        // 10:1 gear box
#define PULSES_PER_REV 65536.0f
#define MAX_MOTOR_SPEED_CMD MAX_JOINT_VEL
#define HOME_ANGLE_ABOVE_HOR 57.2242812f
#define HOME_PULSE_OFFSET_DEFAULT (HOME_ANGLE_ABOVE_HOR * (PULSES_PER_REV / 360.0f) * JOINT_GEAR_RATIO)
// Calibrated offsets from end-stop encoder zero to true kinematic HOME (z = HOME_Z).
// Tune per motor from measured pulses after homing procedure.
#define HOME_PULSE_OFFSET_M1 HOME_PULSE_OFFSET_DEFAULT
#define HOME_PULSE_OFFSET_M2 HOME_PULSE_OFFSET_DEFAULT
#define HOME_PULSE_OFFSET_M3 HOME_PULSE_OFFSET_DEFAULT
// Joint sign calibration between motor raw direction and model direction.
// Set to -1.0f for a motor if commanded Z motion is inverted.
#define ROBOT_JOINT_SIGN_1 1.0f
#define ROBOT_JOINT_SIGN_2 1.0f
#define ROBOT_JOINT_SIGN_3 1.0f

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
// Calculate distance and output component differences. Returns euclidean distance.
float robot_calc_dist(vec3 current, vec3 target, float *out_dx, float *out_dy, float *out_dz);
bool robot_target_in_workspace(vec3 pos);
// Convert between IK angle frame and encoder angle frame (HOME == 0 deg).
bool robot_joint_angles_ik_to_encoder(float q1_ik_deg, float q2_ik_deg, float q3_ik_deg,
                                      float *q1_enc_deg, float *q2_enc_deg, float *q3_enc_deg);
bool robot_joint_angles_encoder_to_ik(float q1_enc_deg, float q2_enc_deg, float q3_enc_deg,
                                      float *q1_ik_deg, float *q2_ik_deg, float *q3_ik_deg);

// Safety/state helpers
void set_idle(robot_t *robot);
void stop_motion(void);
void safety_enter_fault_mode(void);
