#include "motion_execute.h"

#include <math.h>

#include "main.h"
#include "robot_runtime.h"

static volatile uint8_t s_motion_tick_pending = 0U;
static uint32_t s_tick_ms_accum = 0U;
// Set to 1 for qdot telemetry over UART2. Keep 0 in production to save FLASH.
#define MOTION_DEBUG_QDOT 0

// Pre-computed motion profile constants (independent of targets)
static const float s_ramp_time = MAX_CART_VEL / MAX_CART_ACC;
static const float s_ramp_dist = 0.5f * MAX_CART_ACC * (MAX_CART_VEL / MAX_CART_ACC) * (MAX_CART_VEL / MAX_CART_ACC);

static long clamp_speed_cmd(float cmd, bool *out_clamped)
{
  if (out_clamped) *out_clamped = false;
  
  if (cmd > (float)MAX_MOTOR_SPEED_CMD) {
    if (out_clamped) *out_clamped = true;
    return MAX_MOTOR_SPEED_CMD;
  }
  if (cmd < (float)(-MAX_MOTOR_SPEED_CMD)) {
    if (out_clamped) *out_clamped = true;
    return -MAX_MOTOR_SPEED_CMD;
  }
  return (long)lroundf(cmd);
}

static float joint_deg_s_to_speed_cmd(float joint_deg_per_s)
{
  // deg/s -> RPM = deg/s / 6.0, then scale for driver command units.
  const float rpm = (joint_deg_per_s / 6.0f) * JOINT_GEAR_RATIO;
  return rpm * JOINT_SPEED_CMD_SCALE;
}

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
  move_plan *plan = &robot->current_move_plan;

  plan->active = false;
  robot->flag_ready_to_move = false;
  robot->flag_path_done = true;
  robot->current_pos = plan->target_pos;

  robot_runtime_stop_joint_speed();
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
  robot->current_target.t_arrival_s = 5.0f;
}

void motion_execute_plan_strike(robot_t *robot)
{
  robot->current_target.type = TARGET_STRIKE;
  robot->current_target.pos.x += 50.0f;  // mm
  robot->current_target.t_arrival_s = 2.0f;
}

void motion_execute_plan(robot_t *robot)
{
  move_plan *plan = &robot->current_move_plan;
  const vec3 start = robot_get_current_pos();
  const vec3 target = robot->current_target.pos;
  float dx, dy, dz;
  const float D = robot_calc_dist(start, target, &dx, &dy, &dz);

  float t_acc = 0.0f;
  float t_cruise = 0.0f;
  float t_dec = 0.0f;

  robot->flag_path_done = false;
  robot->flag_path_abort = false;

  plan->start_pos = start;
  robot->current_pos = start;
  plan->target_pos = target;
  plan->D = D;
  plan->prev_tick_ms = HAL_GetTick();

  // Seed previous joint state from live encoder readings so the first execute
  // tick can compute qdot immediately (no one-tick startup delay).
  float q1_now, q2_now, q3_now;
  if (robot_runtime_get_joint_angles(&q1_now, &q2_now, &q3_now)) {
    plan->prev_joint_deg[0] = q1_now;
    plan->prev_joint_deg[1] = q2_now;
    plan->prev_joint_deg[2] = q3_now;
    plan->prev_joint_valid = true;
  } else {
    plan->prev_joint_valid = false;
    robot_runtime_send_status("ERR: prev q unset\r\n");
  }

  // 1mm theshold
  if (D > 1.0f) {
    plan->dir.x = dx / D;
    plan->dir.y = dy / D;
    plan->dir.z = dz / D;
  } else {
    plan->dir = (vec3){0.0f, 0.0f, 0.0f};
    plan->t1 = 0.0f;
    plan->t2 = 0.0f;
    plan->t3 = 0.0f;
    plan->T = 0.0f;
    plan->t_start_ms = HAL_GetTick();
    plan->active = false;
    robot->flag_path_done = true;
    return;
  }

  if (D <= 2.0f * s_ramp_dist) {
    t_acc = sqrtf(D / MAX_CART_ACC);
    t_cruise = 0.0f;
    t_dec = t_acc;
  } else {
    t_acc = s_ramp_time;
    t_cruise = (D - 2.0f * s_ramp_dist) / MAX_CART_VEL;
    t_dec = s_ramp_time;
  }

  float t_extra = robot->current_target.t_arrival_s - (t_acc + t_cruise + t_dec);
  if (t_extra < 0.0f) {
    t_extra = 0.0f;
  }

  plan->t1 = t_extra;
  plan->t2 = plan->t1 + t_acc;
  plan->t3 = plan->t2 + t_cruise;
  plan->T = plan->t3 + t_dec;
  plan->t_start_ms = HAL_GetTick();
  plan->active = true;
}

void motion_execute_start(robot_t *robot)
{
  if (!robot->current_move_plan.prev_joint_valid) {
    robot->flag_ready_to_move = false;
    robot_runtime_send_status("PLAN_ABORT: NO_FB\r\n");
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
  const float t_s = ((float)(now_ms - plan->t_start_ms)) * 0.001f;

  if (t_s >= plan->T) {
    motion_finish(robot);
    return;
  }

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

  float q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
  if (!robot_joint_angles_ik_to_encoder(q1_ik, q2_ik, q3_ik, &q1, &q2, &q3)) {
    motion_abort(robot, "PATH_ABORT: FRAME_MAP\r\n");
    return;
  }

  float q1_cont = q1;
  float q2_cont = q2;
  float q3_cont = q3;

  if (plan->prev_joint_valid) {
    const uint32_t dt_ms = now_ms - plan->prev_tick_ms;
    if (dt_ms > 0U) {
      const float q1_prev = plan->prev_joint_deg[0];
      const float q2_prev = plan->prev_joint_deg[1];
      const float q3_prev = plan->prev_joint_deg[2];

      q1_cont = unwrap_deg_near(q1, q1_prev);
      q2_cont = unwrap_deg_near(q2, q2_prev);
      q3_cont = unwrap_deg_near(q3, q3_prev);

      const float dt_s = ((float)dt_ms) * 0.001f;
      const float qdot1 = (q1_cont - q1_prev) / dt_s;
      const float qdot2 = (q2_cont - q2_prev) / dt_s;
      const float qdot3 = (q3_cont - q3_prev) / dt_s;

#if MOTION_DEBUG_QDOT
      char dbg[120];
      snprintf(dbg, sizeof(dbg), "qd=(%ld,%ld,%ld)\r\n",
               (long)(qdot1), (long)(qdot2), (long)(qdot3));
      robot_runtime_send_status(dbg);
#endif

      bool clamped1, clamped2, clamped3;
      const long cmd1 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot1), &clamped1);
      const long cmd2 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot2), &clamped2);
      const long cmd3 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot3), &clamped3);

      // Abort if any motor speed was clamped (path would deviate)
      if (clamped1 || clamped2 || clamped3) {
        motion_abort(robot, "PATH_ABORT: MOTOR_SPEED_EXCEEDED\r\n");
        return;
      }

      robot_runtime_set_joint_speed(cmd1, cmd2, cmd3);
    }
  }

  plan->prev_joint_deg[0] = q1_cont;
  plan->prev_joint_deg[1] = q2_cont;
  plan->prev_joint_deg[2] = q3_cont;
  plan->prev_joint_valid = true;
  plan->prev_tick_ms = now_ms;
  robot->current_pos = setpoint;
}

void motion_execute_stop_all(void)
{
  robot_runtime_stop_joint_speed();
}
