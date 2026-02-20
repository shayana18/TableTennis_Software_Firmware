#include "robot.h"
#include "robot_runtime.h"
#include "motion_execute.h" 
#include <math.h>

#define PI_F 3.14159265358979323846f
#define DTR (PI_F / 180.0f)
#define SQRT3 1.7320508075688772f
#define TAN30 (1.0f / SQRT3)
#define SIN30 0.5f
#define TAN60 SQRT3
#define SIN120 0.8660254037844386f
#define COS120 -0.5f

const vec3 home = {HOME_X, HOME_Y, HOME_Z};

bool robot_get_joint_angles(float *q1_deg, float *q2_deg, float *q3_deg)
{

  long t1, t2, t3;
  if (!q1_deg || !q2_deg || !q3_deg) return false;
  if (!robot_runtime_get_joint_ticks(&t1, &t2, &t3)) return false;

  // Convert encoder pulses to joint angles in degrees, and converted to correct reference frame for kinematics.
  *q1_deg = ((float)t1 * 360.0f / ((float)PULSES_PER_REV * JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN) + ENCODER_ZERO_TO_IK_OFFSET;
  *q2_deg = ((float)t2 * 360.0f / ((float)PULSES_PER_REV * JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN) + ENCODER_ZERO_TO_IK_OFFSET;
  *q3_deg = ((float)t3 * 360.0f / ((float)PULSES_PER_REV * JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN) + ENCODER_ZERO_TO_IK_OFFSET;

  return true;
}

vec3 robot_get_current_pos(void)
{
  static vec3 last_known_pos = {HOME_X, HOME_Y, HOME_Z};
  float q1_enc, q2_enc, q3_enc;
  if (robot_get_joint_angles(&q1_enc, &q2_enc, &q3_enc)) {
    // float q1_ik, q2_ik, q3_ik;
    // robot_joint_angles_encoder_to_ik(q1_enc, q2_enc, q3_enc, &q1_ik, &q2_ik, &q3_ik);
    last_known_pos = FK(q1_enc, q2_enc, q3_enc);
    return last_known_pos;
  }
  // Keep the last valid estimate instead of snapping to HOME; this avoids
  // replaying the same relative-looking move when a read fails.
  robot_runtime_send_status("UNABLE TO READ MOTOR ANGLES. USING LAST KNOWN POSITION.\r\n");
  return last_known_pos;
}

void robot_set_target_from_mail(robot_target_t *dst, const target_t *src)
{
  if (dst == NULL || src == NULL) {
    return;
  }

  dst->type = src->type;
  dst->target_ID = 0.0f;
  dst->pos = src->intercept_pos;
  dst->t_arrival_s = src->intercept_time;
  dst->timestamp = src->timestamp;
}

float robot_calc_dist(vec3 current, vec3 target, float *out_dx, float *out_dy, float *out_dz)
{
  const float x = target.x - current.x;
  const float y = target.y - current.y;
  const float z = target.z - current.z;
  if (out_dx) *out_dx = x;
  if (out_dy) *out_dy = y;
  if (out_dz) *out_dz = z;
  return sqrtf(x * x + y * y + z * z);
}

bool robot_target_in_workspace(vec3 pos)
{
  if (pos.x < LIMIT_NEG_X || pos.x > LIMIT_POS_X) {
    return false;
  }
  if (pos.y < LIMIT_NEG_Y || pos.y > LIMIT_POS_Y) {
    return false;
  }
  if (pos.z < LIMIT_NEG_Z || pos.z > LIMIT_POS_Z) {
    return false;
  }
  return true;
}

// Helper: compute unit vector in XY plane
static void unit_xy_2d(float vx, float vy, float *out_ux, float *out_uy)
{
  float n = hypotf(vx, vy);
  if (n < 1e-9f) {
    *out_ux = 0.0f;
    *out_uy = 0.0f;
  } else {
    *out_ux = vx / n;
    *out_uy = vy / n;
  }
}

// DELTA ROBOT FORWARD KINEMATICS (analytical, matches IK geometry)
vec3 FK(float theta1_deg, float theta2_deg, float theta3_deg)
{
  const float Rb = BASE_RADIUS;
  const float Re = EE_RADIUS;
  const float rf = UPPER_ARM_LENGTH;
  const float re = LOWER_ARM_LENGTH;

  // Convert degrees to radians
  const float th1 = theta1_deg * DTR;
  const float th2 = theta2_deg * DTR;
  const float th3 = theta3_deg * DTR;

  // Base triangle shoulder positions (oriented to match IK)
  const float B1x = 0.0f,           B1y = -Rb;
  const float B2x = Rb * SQRT3 / 2.0f, B2y = Rb / 2.0f;
  const float B3x = -Rb * SQRT3 / 2.0f, B3y = Rb / 2.0f;

  // Unit vectors from origin to each shoulder (in XY)
  float u1x, u1y, u2x, u2y, u3x, u3y;
  unit_xy_2d(B1x, B1y, &u1x, &u1y);
  unit_xy_2d(B2x, B2y, &u2x, &u2y);
  unit_xy_2d(B3x, B3y, &u3x, &u3y);

  // Elbow positions E1, E2, E3 from joint angles
  const float E1x = B1x + rf * cosf(th1) * u1x;
  const float E1y = B1y + rf * cosf(th1) * u1y;
  const float E1z = rf * sinf(th1);

  const float E2x = B2x + rf * cosf(th2) * u2x;
  const float E2y = B2y + rf * cosf(th2) * u2y;
  const float E2z = rf * sinf(th2);

  const float E3x = B3x + rf * cosf(th3) * u3x;
  const float E3y = B3y + rf * cosf(th3) * u3y;
  const float E3z = rf * sinf(th3);

  // End-effector joint offsets (wrist positions relative to EE center)
  const float V1x = 0.0f,            V1y = -Re,                 V1z = 0.0f;
  const float V2x = Re * SQRT3 / 2.0f, V2y = Re / 2.0f,           V2z = 0.0f;
  const float V3x = -Re * SQRT3 / 2.0f, V3y = Re / 2.0f,           V3z = 0.0f;

  // For each constraint || C + Vi - Ei ||^2 = re^2, expand to:
  // x^2 + y^2 + z^2 + 2*(px*x + py*y + pz*z) + c_const = 0
  // where p = V - E and c_const includes all constant terms
  
  #define EXPANDED_TERMS(E_x, E_y, E_z, V_x, V_y, V_z, out_px, out_py, out_pz, out_c) \
  do { \
    float px = (V_x) - (E_x); \
    float py = (V_y) - (E_y); \
    float pz = (V_z) - (E_z); \
    float E_sq = (E_x)*(E_x) + (E_y)*(E_y) + (E_z)*(E_z); \
    float V_sq = (V_x)*(V_x) + (V_y)*(V_y) + (V_z)*(V_z); \
    float EV = (E_x)*(V_x) + (E_y)*(V_y) + (E_z)*(V_z); \
    float c_val = E_sq + V_sq - 2.0f*EV - re*re; \
    (out_px) = px; (out_py) = py; (out_pz) = pz; (out_c) = c_val; \
  } while(0)

  float p1x, p1y, p1z, c1;
  float p2x, p2y, p2z, c2;
  float p3x, p3y, p3z, c3;

  EXPANDED_TERMS(E1x, E1y, E1z, V1x, V1y, V1z, p1x, p1y, p1z, c1);
  EXPANDED_TERMS(E2x, E2y, E2z, V2x, V2y, V2z, p2x, p2y, p2z, c2);
  EXPANDED_TERMS(E3x, E3y, E3z, V3x, V3y, V3z, p3x, p3y, p3z, c3);

  // Subtract equation 1 from 2 and 3 to get linear system:
  // (p2 - p1) · [x,y,z] = (c1 - c2)/2
  // (p3 - p1) · [x,y,z] = (c1 - c3)/2
  float A2 = p2x - p1x;
  float B2 = p2y - p1y;
  float C2 = p2z - p1z;
  float D2 = (c1 - c2) / 2.0f;

  float A3 = p3x - p1x;
  float B3_var = p3y - p1y;
  float C3 = p3z - p1z;
  float D3 = (c1 - c3) / 2.0f;

  // Solve for x, y in terms of z: x = x0_0 + x0_1*z, y = y0_0 + y0_1*z
  float det = A2 * B3_var - A3 * B2;
  if (fabsf(det) < 1e-9f) {
    // Singular system
    return (vec3){0.0f, 0.0f, 0.0f};
  }

  float x0_0 = (D2 * B3_var - D3 * B2) / det;
  float x0_1 = (-C2 * B3_var + C3 * B2) / det;
  float y0_0 = (A2 * D3 - A3 * D2) / det;
  float y0_1 = (-A2 * C3 + A3 * C2) / det;

  // Plug into equation 1 to get quadratic in z:
  // (x0_0 + x0_1*z)^2 + (y0_0 + y0_1*z)^2 + z^2 + 2*(...)*z + ... = 0
  float qa = x0_1*x0_1 + y0_1*y0_1 + 1.0f;
  float qb = 2.0f*(x0_0*x0_1 + y0_0*y0_1) + 2.0f*(p1x*x0_1 + p1y*y0_1 + p1z);
  float qc = x0_0*x0_0 + y0_0*y0_0 + 2.0f*(p1x*x0_0 + p1y*y0_0) + c1;

  float disc = qb*qb - 4.0f*qa*qc;
  if (disc < 0.0f) {
    // No real solution
    return (vec3){0.0f, 0.0f, 0.0f};
  }

  // Choose the root matching the "z negative = down" branch (same as IK)
  float sqrt_disc = sqrtf(disc);
  float z0 = (-qb - sqrt_disc) / (2.0f * qa);

  float x0 = x0_0 + x0_1 * z0;
  float y0 = y0_0 + y0_1 * z0;

  // Align FK output with IK/world frame convention used by IK(target) in this
  // codebase (without this, FK(IK(p)) mirrors X).
  return (vec3){-x0, y0, z0};
}

// Helper function for IK: solves one arm in YZ plane.
static int calc_angleYZ(float x0, float y0, float z0, float *theta)
{
  if (fabsf(z0) < 1.0f) {
    return -1;
  }

  const float y1 = -BASE_RADIUS;
  const float y0p = y0 - EE_RADIUS;

  const float a = (x0 * x0 + y0p * y0p + z0 * z0 + UPPER_ARM_LENGTH * UPPER_ARM_LENGTH
      - LOWER_ARM_LENGTH * LOWER_ARM_LENGTH - y1 * y1) / (2.0f * z0);
  const float b = (y1 - y0p) / z0;

  const float term = a + b * y1;
  const float d = -(term * term) + UPPER_ARM_LENGTH * (b * b * UPPER_ARM_LENGTH + UPPER_ARM_LENGTH);
  if (d < 0.0f) {
    return -1;
  }

  const float sqrt_d = sqrtf(d);
  const float yj = (y1 - a * b - sqrt_d) / (b * b + 1.0f);
  const float zj = a + b * yj;

  *theta = -PI_F - atan2f(zj, (yj - y1));
  *theta *= (180.0f / PI_F);
  return 0;
}

// DELTA ROBOT INVERSE KINEMATICS
int IK(float x0, float y0, float z0, float *t1, float *t2, float *t3)
{
  float theta1;
  float theta2;
  float theta3;

  if (calc_angleYZ(x0, y0, z0, &theta1) != 0) {
    return -1;
  }

  const float x1 = x0 * COS120 - y0 * SIN120;
  const float y1 = y0 * COS120 + x0 * SIN120;
  if (calc_angleYZ(x1, y1, z0, &theta2) != 0) {
    return -1;
  }

  const float x2 = x0 * COS120 + y0 * SIN120;
  const float y2 = y0 * COS120 - x0 * SIN120;
  if (calc_angleYZ(x2, y2, z0, &theta3) != 0) {
    return -1;
  }

  *t1 = theta1;
  *t2 = theta2;
  *t3 = theta3;
  return 0;
}

void stop_motion(void)
{
  // Motor stop commands are issued by FSM when IO context is available.
}

void set_idle(robot_t *robot)
{
  motion_execute_stop_all();
  if (robot != NULL) {
    robot->flag_ready_to_move = false;
  }
}

void safety_enter_fault_mode(void)
{
  // TODO: add platform specific fault outputs and motor disable handling.
}

