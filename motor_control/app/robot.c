#include "robot.h"

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

vec3 robot_get_current_pos(void)
{
  // TODO: replace with encoder -> joint angle -> FK path.
  return home;
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

float robot_calc_dist(vec3 current, vec3 target)
{
  const float x = target.x - current.x;
  const float y = target.y - current.y;
  const float z = target.z - current.z;
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

// DELTA ROBOT FORWARD KINEMATICS
vec3 FK(float motor_q1, float motor_q2, float motor_q3)
{
  const float f = BASE_RADIUS * 2.0f;
  const float e = EE_RADIUS * 2.0f;
  const float rf = UPPER_ARM_LENGTH;
  const float re = LOWER_ARM_LENGTH;

  // Convert to radians
  const float t1 = motor_q1 * DTR;
  const float t2 = motor_q2 * DTR;
  const float t3 = motor_q3 * DTR;

  const float t = (f - e) * TAN30 * 0.5f;

  const float y1 = -(t + rf * cosf(t1));
  const float z1 = -rf * sinf(t1);

  const float y2 = (t + rf * cosf(t2)) * SIN30;
  const float x2 = y2 * TAN60;
  const float z2 = -rf * sinf(t2);

  const float y3 = (t + rf * cosf(t3)) * SIN30;
  const float x3 = -y3 * TAN60;
  const float z3 = -rf * sinf(t3);

  const float dnm = (y2 - y1) * x3 - (y3 - y1) * x2;

  if (fabsf(dnm) < 1e-9f) {
    return (vec3){0.0f, 0.0f, 0.0f};
  }

  const float w1 = y1 * y1 + z1 * z1;
  const float w2 = x2 * x2 + y2 * y2 + z2 * z2;
  const float w3 = x3 * x3 + y3 * y3 + z3 * z3;

  const float a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1);
  const float b1 = -0.5f * ((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1));

  const float a2 = -(z2 - z1) * x3 + (z3 - z1) * x2;
  const float b2 = 0.5f * ((w2 - w1) * x3 - (w3 - w1) * x2);

  const float A = a1 * a1 + a2 * a2 + dnm * dnm;
  const float B = 2.0f * (a1 * b1 + a2 * b2 + dnm * dnm * z1);
  const float C = b1 * b1 + b2 * b2 + dnm * dnm * (z1 * z1 - re * re);

  float D = B * B - 4.0f * A * C;
  if (D < 0.0f) {
    if (D > -1e-6f) {
      D = 0.0f;
    } else {
      return (vec3){0.0f, 0.0f, 0.0f};
    }
  }

  const float sqrtD = sqrtf(D);
  const float z = -0.5f * (B + sqrtD) / A;
  const float x = (a1 * z + b1) / dnm;
  const float y = (a2 * z + b2) / dnm;

  return (vec3){x, y, z};
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
  stop_motion();
  if (robot != NULL) {
    robot->flag_ready_to_move = false;
  }
}

void safety_enter_fault_mode(void)
{
  // TODO: add platform specific fault outputs and motor disable handling.
}
