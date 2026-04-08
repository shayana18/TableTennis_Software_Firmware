#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "shared_types.h"

// Robot motion limits
#define MAX_JOINT_ANGLE_LIMIT ENCODER_ZERO_TO_IK_OFFSET // degrees
#define MIN_JOINT_ANGLE_LIMIT -75.0f  // Low stop limit is 80 deg
#define MAX_JOINT_VEL 3000.0L      // RPM, Conservative values
#define MAX_JOINT_ACC 1000.0L     // RPM/s, Conservative Values

#define MAX_CART_VEL 2000.0f     // mm/s Default: 4000
#define MAX_CART_ACC 2000.0f    // mm/s^2  default: 20000

// Home position (mm)
#define HOME_X 0.0f
#define HOME_Y 0.0f
#define HOME_Z -900.0f
#define HOME_TIME 2.0f

// Delta geometry (mm)
#define BASE_RADIUS 165.0f
#define EE_RADIUS 50.0f
#define UPPER_ARM_LENGTH 350.0f
#define LOWER_ARM_LENGTH 1000.0f

// Workspace bounds (mm) - Elliptic Cylinder
#define ELLIPSE_RADIUS_X 600.0f
#define ELLIPSE_RADIUS_Y 550.0f
#define LIMIT_POS_Z -650.0f
#define LIMIT_NEG_Z -1050.0f

// Paddle Workspace Bounds (mm)
#define ELLIPSE_RADIUS_X_PADDLE 820.0f
#define ELLIPSE_RADIUS_Y_PADDLE 550.0f

// Motion execution configuration
#define MOTION_EXECUTE_PERIOD_MS 10U
#define JOINT_GEAR_RATIO 10.0f        // 10:1 gear box
#define PULSES_PER_REV 65536.0f
#define MAX_MOTOR_SPEED_CMD MAX_JOINT_VEL
#define ENCODER_ZERO_TO_IK_OFFSET 57.2242812f  // deg
#define HOME_PULSE_OFFSET_DEFAULT (ENCODER_ZERO_TO_IK_OFFSET * (PULSES_PER_REV / 360.0f) * JOINT_GEAR_RATIO)
// Calibrated offsets from end-stop encoder zero to true kinematic HOME (z = HOME_Z).
// Tune per motor from measured pulses after homing procedure.
#define HOME_PULSE_OFFSET_M1 HOME_PULSE_OFFSET_DEFAULT
#define HOME_PULSE_OFFSET_M2 HOME_PULSE_OFFSET_DEFAULT
#define HOME_PULSE_OFFSET_M3 HOME_PULSE_OFFSET_DEFAULT
// Joint sign calibration between motor raw direction and model direction.
// Set to -1.0f for a motor if commanded Z motion is inverted.
#define ROBOT_JOINT_SIGN -1.0f
#define Q_TOLERANCE 0.5 // degrees

#define BUFFER_TIME 0.05f // seconds, Time buffer to account for communication and execution delays when planning to target arrival time.


// Motor IDs on the daisy-chain
#define ROBOT_MOTOR_1_ID 2
#define ROBOT_MOTOR_2_ID 3
#define ROBOT_MOTOR_3_ID 4

extern const vec3 home;

typedef struct {
  target_type type;
  float target_ID;
  vec3 pos;
  vec3 ball_vel;       // (mm/s) ball velocity at intercept
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
  float last_feedback_deg[3];
  bool feedback_valid;
  uint8_t feedback_miss_ticks;
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
int robot_delta_inv_jacobian(const float theta_deg[3], vec3 C, float Jinv[3][3]);

// Robot helpers
void robot_set_target_from_mail(robot_target_t *dst, const target_t *src);
// Calculate distance and output component differences. Returns euclidean distance.
float robot_calc_dist(vec3 current, vec3 target, float *out_dx, float *out_dy, float *out_dz);
bool robot_target_in_workspace(vec3 pos);

bool robot_get_joint_angles(float *q1_deg, float *q2_deg, float *q3_deg);

// Safety/state helpers
void set_idle(robot_t *robot);
void stop_motion(void);
void safety_enter_fault_mode(void);

void print_joint_angles();
