#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "shared_types.h"

// Constants
#define GRAVITY 9810.0f // mm/s^2
#define PI_F 3.14159265358979323846f
#define DTR (PI_F / 180.0f)
#define SQRT3 1.7320508075688772f
#define TAN30 (1.0f / SQRT3)
#define SIN30 0.5f
#define TAN60 SQRT3
#define SIN120 0.8660254037844386f
#define COS120 -0.5f
#define HALF_PI_F (PI_F / 2.0f)

// Robot motion limits
#define MAX_JOINT_ANGLE_LIMIT ENCODER_ZERO_TO_IK_OFFSET // degrees
#define MIN_JOINT_ANGLE_LIMIT -79.0f  // Low stop limit is 80 deg
#define MAX_JOINT_VEL 3000.0L      // RPM, Conservative values
#define MAX_JOINT_ACC 10000.0L     // RPM/s, Conservative Values

#define MAX_CART_VEL 4000.0f     // mm/s Default: 4000
#define MAX_CART_ACC 17000.0f    // mm/s^2  default: 20000
#define HOME_CART_ACC 4000.0f   // mm/s^2 slower home acceleration to reduce oscillation

#define MAX_STRIKE_VEL MAX_CART_VEL    // mm/s

// Home position (mm)
#define HOME_X 0.0f
#define HOME_Y 0.0f
#define HOME_Z -880.0f
#define HOME_TIME 2.0f

// Striking Parameters
#define STRIKE_TARGET_X 0.0f
#define STRIKE_TARGET_Y 2055.0f
#define STRIKE_TARGET_Z -1150.0f
#define STRIKE_BUFFER_DIST 0.0f //mm Additional distance to account for strike execution delay and ball contact
#define STRIKE_Z_APEX_CLEARANCE 100.0f 
#define RESTITUTION 0.90f
#define STRIKE_POWER_GAIN 2.5f
#define STRIKE_MIN_COMMAND_SPEED 2000.0f // (mm/s) minimum commanded paddle speed when striking
#define STRIKE_SWEEP_DIST_GAIN 1.0f
#define STRIKE_SWEEP_EXTRA_DIST 50.0f // (mm) adds extra counter-sweep coverage on top of stopping distance
#define STRIKE_WINDUP_DIST_GAIN 1.0f // (>1) enlarge pre-contact travel for stronger impact speed
#define STRIKE_FOLLOWTHROUGH_DIST_GAIN 1.0f // (<1) keep post-contact travel shorter than windup
#define STRIKE_TAU_WINDOW_GAIN 1.35f
#define STRIKE_TAU_WINDOW_MIN 0.015f // (s)
#define STRIKE_TAU_WINDOW_MAX 0.180f // (s)
#define STRIKE_TAU_POS_SCALE 1.60f
#define STRIKE_TAU_NEG_SCALE 0.70f
#define STRIKE_CONTACT_LEAD_TIME 0.080f // (s) schedule impact this much earlier to compensate latency
#define STRIKE_YAW_BIAS_DEG 0.0f // (deg) positive bias to steer returns more toward opponent center

// Delta geometry (mm)
#define BASE_RADIUS 165.0f
#define EE_RADIUS 50.0f
#define UPPER_ARM_LENGTH 350.0f
#define LOWER_ARM_LENGTH 1000.0f
#define PADDLE_X_OFFSET 206.0f // mm
#define PADDLE_Y_OFFSET 34.0f // mm
#define PADDLE_OFFSET_Z -88.0f   // mm

// PADDLE Workspace bounds (mm) - (Paddle)  Elliptic Cylinder
#define PADDLE_ELLIPSE_RADIUS_X 790.0f
#define PADDLE_ELLIPSE_RADIUS_Y 540.0f
#define PADDLE_LIMIT_POS_Z -800.0f
#define PADDLE_LIMIT_NEG_Z -1020.0f
#define PADDLE_STARTING_OFFSET_DEG 230.0f // deg, angle relative to horizontal
#define PADDLE_TICKS_PER_DEG ((PULSES_PER_REV * PADDLE_GEAR_RATIO) / 360.0f)
#define NET_Z_TOP -1000.0f

// ROBOT EE Workspace bounds (mm) - (Robot EE)  Elliptic Cylinder
#define ROBOT_EE_ELLIPSE_RADIUS_X   (PADDLE_ELLIPSE_RADIUS_X - PADDLE_X_OFFSET)
#define ROBOT_EE_ELLIPSE_RADIUS_Y   (PADDLE_ELLIPSE_RADIUS_Y)
#define ROBOT_EE_LIMIT_POS_Z        (PADDLE_LIMIT_POS_Z - PADDLE_OFFSET_Z)
#define ROBOT_EE_LIMIT_NEG_Z        (PADDLE_LIMIT_NEG_Z - PADDLE_OFFSET_Z)

// Interception Workspace bounds (mm)
#define INTERCEPTION_ELLIPSE_RADIUS_X (PADDLE_ELLIPSE_RADIUS_X - sqrtf((MAX_STRIKE_VEL * MAX_STRIKE_VEL) / (2.0f * MAX_CART_ACC)) - STRIKE_BUFFER_DIST)
#define INTERCEPTION_ELLIPSE_RADIUS_Y (PADDLE_ELLIPSE_RADIUS_Y - sqrtf((MAX_STRIKE_VEL * MAX_STRIKE_VEL) / (2.0f * MAX_CART_ACC)) - STRIKE_BUFFER_DIST)
#define INTERCEPTION_LIMIT_POS_Z      (PADDLE_LIMIT_POS_Z)
#define INTERCEPTION_LIMIT_NEG_Z      (PADDLE_LIMIT_NEG_Z)

// Paddle Workspace Bounds (mm)
#define ELLIPSE_RADIUS_X_PADDLE 820.0f
#define ELLIPSE_RADIUS_Y_PADDLE 550.0f

// Motion execution configuration
#define MOTION_EXECUTE_PERIOD_MS 4U
#define JOINT_GEAR_RATIO 10.0f        // 10:1 gear box
#define PADDLE_GEAR_RATIO 10.0f       // Paddle axis gearbox ratio (motor rev : paddle rev)
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
#define ROBOT_MOTOR_1_ID 1 // PADDLE AXIS!
#define ROBOT_MOTOR_2_ID 2
#define ROBOT_MOTOR_3_ID 3
#define ROBOT_MOTOR_4_ID 4

extern const vec3 home;

typedef struct {
  target_type type;
  float target_ID;
  vec3 pos;
  vec3 vel;
  float t_arrival_s;
  float timestamp;
  uint32_t received_time;
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
  float max_cart_vel;
  float max_cart_acc;
  float yaw_angle_deg;
  bool ballistic_track;
  vec3 ballistic_intercept_pos;
  vec3 ballistic_intercept_vel;
  float ballistic_tau_window_s;
  float strike_contact_progress;

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
  move_plan strike_move_plan;

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
bool robot_EE_in_workspace(vec3 pos);
bool robot_target_in_workspace(vec3 pos);

bool robot_get_joint_angles(float *q1_deg, float *q2_deg, float *q3_deg);

// Safety/state helpers
void set_idle(robot_t *robot);
void stop_motion(void);
void safety_enter_fault_mode(void);

void print_joint_angles();
