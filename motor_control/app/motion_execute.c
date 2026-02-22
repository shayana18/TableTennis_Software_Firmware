#include "motion_execute.h"

#include <math.h>

#include "main.h"
#include "robot_runtime.h"

static volatile uint8_t s_motion_tick_pending = 0U;
static uint32_t s_tick_ms_accum = 0U;

#if 0
// Legacy velocity-control configuration and helpers (disabled).
// Re-enable by moving this block out of #if 0 and switching tick path below.
#define MOTION_DEBUG_QDOT 0
#define MOTION_JOINT_CORRECTION_KP 1.35f
#define MOTION_MAX_FB_MISS_TICKS 3U

static bool speed_cmd_exceeds_limit(float cmd)
{
  return (fabsf(cmd) > (float)MAX_MOTOR_SPEED_CMD);
}

static float joint_deg_s_to_speed_cmd(float joint_deg_per_s)
{
  // deg/s -> RPM = deg/s / 6.0, then scale for driver command units.
  const float rpm = (joint_deg_per_s / 6.0f);
  return rpm * JOINT_GEAR_RATIO;
}

static float motion_profile_speed(const move_plan *plan, float t_s)
{
  const float t_acc = plan->t2 - plan->t1;
  const float a = MAX_CART_ACC;

  if (t_s <= plan->t1) {
    return 0.0f;
  }

  if (t_s < plan->t2) {
    return a * (t_s - plan->t1);
  }

  const float v_peak = a * t_acc;
  if (t_s < plan->t3) {
    return v_peak;
  }

  if (t_s < plan->T) {
    return v_peak - a * (t_s - plan->t3);
  }

  if (t_s >= plan->T) {
    return 0.0f;
  }

  return 0.0f;
}
#endif

// Pre-computed motion profile constants (independent of targets)
static const float s_ramp_time = MAX_CART_VEL / MAX_CART_ACC;
static const float s_ramp_dist = 0.5f * MAX_CART_ACC * (MAX_CART_VEL / MAX_CART_ACC) * (MAX_CART_VEL / MAX_CART_ACC);

static float unwrap_deg_near(float angle_deg, float ref_deg)
{
  float out = angle_deg;
  float delta = out - ref_deg;

  while (delta > 180.0f) {
    out -= 360.0f;
    delta -= 360.0f;
  }
  while (delta < -180.0f) {
    out += 360.0f;
    delta += 360.0f;
  }

  return out;
}

static float motion_profile_distance(const move_plan *plan, float t_s)
{
  const float t_acc = plan->t2 - plan->t1;
  const float t_cruise = plan->t3 - plan->t2;
  const float a = MAX_CART_ACC;

  if (t_s <= plan->t1) {
    return 0.0f;
  }

  if (t_s < plan->t2) {
    const float dt = t_s - plan->t1;
    return 0.5f * a * dt * dt;
  }

  const float v_peak = a * t_acc;
  const float d_acc = 0.5f * a * t_acc * t_acc;

  if (t_s < plan->t3) {
    const float dt = t_s - plan->t2;
    return d_acc + v_peak * dt;
  }

  if (t_s < plan->T) { 
  const float dt = t_s - plan->t3;
  const float d_cruise = v_peak * t_cruise;
  return d_acc + d_cruise + v_peak * dt - 0.5f * a * dt * dt;
  }

  return plan->D;
}

static void motion_abort(robot_t *robot, const char *reason)
{
  move_plan *plan = &robot->current_move_plan;

  plan->active = false;
  robot->flag_ready_to_move = false;
  robot->flag_path_abort = true;

  robot_runtime_stop_joint_speed();
  if (reason != NULL) {
    robot_runtime_send_status(reason);
  }
}

static void motion_finish(robot_t *robot)
{
  // Legacy velocity-mode stop behavior:
  // robot_runtime_stop_joint_speed();

  move_plan *plan = &robot->current_move_plan;

  plan->active = false;
  robot->flag_ready_to_move = false;
  robot->flag_path_done = true;
  robot->current_pos = plan->target_pos;

}

void motion_execute_reset_scheduler(void)
{
  s_motion_tick_pending = 0U;
  s_tick_ms_accum = 0U;
}

void motion_execute_on_timer_tick(void)
{
  s_tick_ms_accum++;
  if (s_tick_ms_accum >= MOTION_EXECUTE_PERIOD_MS) {
    s_tick_ms_accum = 0U;
    s_motion_tick_pending = 1U;
  }
}

bool motion_execute_consume_tick_due(void)
{
  if (s_motion_tick_pending == 0U) {
    return false;
  }
  s_motion_tick_pending = 0U;
  return true;
}

bool motion_execute_safety_check_joint_limits(float q1_deg, float q2_deg, float q3_deg)
{
  // Check if all joint angles are within the configured limits.
  if (q1_deg < MIN_JOINT_ANGLE_LIMIT || q1_deg > MAX_JOINT_ANGLE_LIMIT) {
    return false;
  }
  if (q2_deg < MIN_JOINT_ANGLE_LIMIT || q2_deg > MAX_JOINT_ANGLE_LIMIT) {
    return false;
  }
  if (q3_deg < MIN_JOINT_ANGLE_LIMIT || q3_deg > MAX_JOINT_ANGLE_LIMIT) {
    return false;
  }
  return true;
}

void motion_execute_make_home_target(robot_t *robot)
{
  robot->current_target.type = TARGET_HOME;
  robot->current_target.pos = home;
  robot->current_target.t_arrival_s = HOME_TIME;
}

void motion_execute_plan_strike(robot_t *robot)
{
  // Temporary strike motion
  robot->current_target.type = TARGET_STRIKE;
  robot->current_target.pos.x += 50.0f;  // mm
  robot->current_target.t_arrival_s = 2.0f;
}

void motion_execute_plan(robot_t *robot)
{
  move_plan *plan = &robot->current_move_plan;
  const uint32_t now_ms = HAL_GetTick();
  const vec3 start = robot_get_current_pos();
  const vec3 target = robot->current_target.pos;
  float dx, dy, dz;
  const float D = robot_calc_dist(start, target, &dx, &dy, &dz);

  robot->flag_path_done = false;
  robot->flag_path_abort = false;

  plan->start_pos = start;
  robot->current_pos = start;
  plan->target_pos = target;
  plan->D = D;
  plan->prev_tick_ms = now_ms;


  if (D <= 1.0f) {
    plan->dir = (vec3){0.0f, 0.0f, 0.0f};
    plan->t1 = 0.0f;
    plan->t2 = 0.0f;
    plan->t3 = 0.0f;
    plan->T = 0.0f;
    plan->t_start_ms = now_ms;
    plan->active = false;
    robot->flag_path_done = true;
    robot_runtime_send_status("No move made, target distance is <1mm\r\n");
    return;
  }

  // Seed previous joint state from live encoder readings so unwrap/reference
  // logic has a valid starting point on the first execute tick.
  float q1_now, q2_now, q3_now;
  if (robot_get_joint_angles(&q1_now, &q2_now, &q3_now)) {
    plan->prev_joint_deg[0] = q1_now;
    plan->prev_joint_deg[1] = q2_now;
    plan->prev_joint_deg[2] = q3_now;
    plan->prev_joint_valid = true;
    plan->last_feedback_deg[0] = q1_now;
    plan->last_feedback_deg[1] = q2_now;
    plan->last_feedback_deg[2] = q3_now;
    plan->feedback_valid = true;
    plan->feedback_miss_ticks = 0U;
  } else {
    plan->prev_joint_valid = false;
    plan->feedback_valid = false;
    plan->feedback_miss_ticks = 0U;
    robot_runtime_send_status("ERR: prev q unset\r\n");
    return;
  }

  // Planning

  plan->dir.x = dx / D;
  plan->dir.y = dy / D;
  plan->dir.z = dz / D;

  float t_acc;
  float t_cruise;
  if (D <= 2.0f * s_ramp_dist) {
    t_acc = sqrtf(D / MAX_CART_ACC);
    t_cruise = 0.0f;
  } else {
    t_acc = s_ramp_time;
    t_cruise = (D - 2.0f * s_ramp_dist) / MAX_CART_VEL;
  }

  const float t_move = (2.0f * t_acc) + t_cruise;
  float t_extra = robot->current_target.t_arrival_s - t_move;
  if (t_extra < 0.0f) {
    t_extra = 0.0f;
  }

  plan->t1 = t_extra;
  plan->t2 = plan->t1 + t_acc;
  plan->t3 = plan->t2 + t_cruise;
  plan->T = plan->t3 + t_acc;
  plan->t_start_ms = now_ms;
  plan->active = true;
}

void motion_execute_start(robot_t *robot)
{
  if (!robot->current_move_plan.prev_joint_valid) {
    robot->flag_ready_to_move = false;
    robot_runtime_send_status("PLAN_ABORT: NO_FeedBack\r\n");
    return;
  }
  robot->flag_ready_to_move = true;
}

void motion_execute_tick(robot_t *robot)
{
  move_plan *plan = &robot->current_move_plan;

  if (!plan->active || !robot->flag_ready_to_move) {
    return;
  }

  const uint32_t now_ms = HAL_GetTick();
  // I 
  const float t_s = ((float)(now_ms - plan->t_start_ms)) * 0.001f;

  float s = motion_profile_distance(plan, t_s);
  if (s < 0.0f) {
    s = 0.0f;
  }
  if (s > plan->D) {
    s = plan->D;
  }

  vec3 setpoint = {
      plan->start_pos.x + plan->dir.x * s,
      plan->start_pos.y + plan->dir.y * s,
      plan->start_pos.z + plan->dir.z * s,
  };

  float q1_ik;
  float q2_ik;
  float q3_ik;

  if (IK(setpoint.x, setpoint.y, setpoint.z, &q1_ik, &q2_ik, &q3_ik) != 0) {
    motion_abort(robot, "PATH_ABORT: IK_FAIL\r\n");
    return;
  }

  if (plan->prev_joint_valid) {
    q1_ik = unwrap_deg_near(q1_ik, plan->prev_joint_deg[0]);
    q2_ik = unwrap_deg_near(q2_ik, plan->prev_joint_deg[1]);
    q3_ik = unwrap_deg_near(q3_ik, plan->prev_joint_deg[2]);
  }

  if (!motion_execute_safety_check_joint_limits(q1_ik, q2_ik, q3_ik) &&
      (robot->current_target.type != TARGET_HOME)) {
    motion_abort(robot, "PATH_ABORT: ROBOT JOINT LIMITS EXCEEDED\r\n");
    return;
  }

  // velocity-control stuff (commented out)
/*
  float q_diff_1, q_diff_2, q_diff_3;
  if (plan->prev_joint_valid) {
    float q1_prev, q2_prev, q3_prev;
    if (robot_get_joint_angles(&q1_prev, &q2_prev, &q3_prev)) {
      plan->last_feedback_deg[0] = q1_prev;
      plan->last_feedback_deg[1] = q2_prev;
      plan->last_feedback_deg[2] = q3_prev;
      plan->feedback_valid = true;
      plan->feedback_miss_ticks = 0U;
    } else if (plan->feedback_valid &&
               (plan->feedback_miss_ticks < MOTION_MAX_FB_MISS_TICKS)) {
      plan->feedback_miss_ticks++;
      q1_prev = plan->prev_joint_deg[0];
      q2_prev = plan->prev_joint_deg[1];
      q3_prev = plan->prev_joint_deg[2];
    } else {
      robot_runtime_send_status("ERR: q read failed\r\n");
      motion_abort(robot, "PATH_ABORT: ENCODER_READ_FAIL\r\n");
      return;
    }

    if (!motion_execute_safety_check_joint_limits(q1_prev, q2_prev, q3_prev) &&
        !(robot->current_target.type == TARGET_HOME)) {
      motion_abort(robot, "PATH_ABORT: ROBOT JOINT LIMITS EXCEEDED\r\n");
      return;
    }

    const float v_cart = motion_profile_speed(plan, t_s);
    const vec3 c_dot = {
        plan->dir.x * v_cart,
        plan->dir.y * v_cart,
        plan->dir.z * v_cart,
    };

    const float theta_deg[3] = {q1_prev, q2_prev, q3_prev};
    float Jinv[3][3];
    if (robot_delta_inv_jacobian(theta_deg, setpoint, Jinv) != 0) {
      motion_abort(robot, "PATH_ABORT: JAC_SINGULAR\r\n");
      return;
    }

    float qdot1 = Jinv[0][0] * c_dot.x + Jinv[0][1] * c_dot.y + Jinv[0][2] * c_dot.z;
    float qdot2 = Jinv[1][0] * c_dot.x + Jinv[1][1] * c_dot.y + Jinv[1][2] * c_dot.z;
    float qdot3 = Jinv[2][0] * c_dot.x + Jinv[2][1] * c_dot.y + Jinv[2][2] * c_dot.z;

    q1_ik = unwrap_deg_near(q1_ik, q1_prev);
    q2_ik = unwrap_deg_near(q2_ik, q2_prev);
    q3_ik = unwrap_deg_near(q3_ik, q3_prev);

    q_diff_1 = q1_ik - q1_prev;
    q_diff_2 = q2_ik - q2_prev;
    q_diff_3 = q3_ik - q3_prev;

    qdot1 += MOTION_JOINT_CORRECTION_KP * q_diff_1;
    qdot2 += MOTION_JOINT_CORRECTION_KP * q_diff_2;
    qdot3 += MOTION_JOINT_CORRECTION_KP * q_diff_3;

    const float q1_cmd = joint_deg_s_to_speed_cmd(qdot1);
    const float q2_cmd = joint_deg_s_to_speed_cmd(qdot2);
    const float q3_cmd = joint_deg_s_to_speed_cmd(qdot3);

    const bool exceeded1 = speed_cmd_exceeds_limit(q1_cmd);
    const bool exceeded2 = speed_cmd_exceeds_limit(q2_cmd);
    const bool exceeded3 = speed_cmd_exceeds_limit(q3_cmd);

    if (exceeded1 || exceeded2 || exceeded3) {
      if (exceeded1) {
        robot_runtime_send_status("CMD1 EXCEEDED\r\n");
      } else if (exceeded2) {
        robot_runtime_send_status("CMD2 EXCEEDED\r\n");
      } else if (exceeded3) {
        robot_runtime_send_status("CMD3 EXCEEDED\r\n");
      }
      motion_abort(robot, "PATH_ABORT: MOTOR_SPEED_EXCEEDED\r\n");
      return;
    }

    const long cmd1 = (long)lroundf(q1_cmd);
    const long cmd2 = (long)lroundf(q2_cmd);
    const long cmd3 = (long)lroundf(q3_cmd);
    robot_runtime_set_joint_speed(cmd1, cmd2, cmd3);
  }
*/

  robot_runtime_set_joint_position_abs_deg(q1_ik, q2_ik, q3_ik);

  plan->prev_joint_deg[0] = q1_ik;
  plan->prev_joint_deg[1] = q2_ik;
  plan->prev_joint_deg[2] = q3_ik;
  plan->prev_joint_valid = true;
  plan->prev_tick_ms = now_ms;
  robot->current_pos = setpoint;

  if (t_s >= plan->T) {
    motion_finish(robot);
    return;
  }
}



void motion_execute_stop_all(void)
{
  // Legacy velocity-mode behavior:
  // robot_runtime_stop_joint_speed();

  // In position-control mode we avoid overriding the final absolute target.
}
