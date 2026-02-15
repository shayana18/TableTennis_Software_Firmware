#include "motion_execute.h"

#include <math.h>

#include "main.h"
#include "robot_runtime.h"

static volatile uint8_t s_motion_tick_pending = 0U;
static uint32_t s_tick_ms_accum = 0U;

static long clamp_speed_cmd(float cmd)
{
  if (cmd > (float)MAX_MOTOR_SPEED_CMD) {
    return MAX_MOTOR_SPEED_CMD;
  }
  if (cmd < (float)(-MAX_MOTOR_SPEED_CMD)) {
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

bool motion_execute_safety_check_joint_limits(void)
{
  // TODO: replace with live encoder-angle checks per joint.
  return true;
}

void motion_execute_make_home_target(robot_t *robot)
{
  robot->current_target.type = TARGET_HOME;
  robot->current_target.pos = home;
  robot->current_target.t_arrival_s = 3.0f;
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
  const vec3 start = robot->current_pos;
  const vec3 target = robot->current_target.pos;
  const float D = robot_calc_dist(start, target);

  const float dx = target.x - start.x;
  const float dy = target.y - start.y;
  const float dz = target.z - start.z;

  const float ramp_time = MAX_CART_VEL / MAX_CART_ACC;
  const float ramp_dist = 0.5f * MAX_CART_ACC * ramp_time * ramp_time;

  float t_acc = 0.0f;
  float t_cruise = 0.0f;
  float t_dec = 0.0f;

  robot->flag_path_done = false;
  robot->flag_path_abort = false;

  plan->start_pos = start;
  plan->target_pos = target;
  plan->D = D;
  plan->prev_joint_valid = false;
  plan->prev_tick_ms = HAL_GetTick();

  if (D > 1e-6f) {
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

  if (D <= 2.0f * ramp_dist) {
    t_acc = sqrtf(D / MAX_CART_ACC);
    t_cruise = 0.0f;
    t_dec = t_acc;
  } else {
    t_acc = ramp_time;
    t_cruise = (D - 2.0f * ramp_dist) / MAX_CART_VEL;
    t_dec = t_acc;
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

  float q1;
  float q2;
  float q3;
  if (IK(setpoint.x, setpoint.y, setpoint.z, &q1, &q2, &q3) != 0) {
    motion_abort(robot, "PATH_ABORT: IK_FAIL\r\n");
    return;
  }

  if (plan->prev_joint_valid) {
    const uint32_t dt_ms = now_ms - plan->prev_tick_ms;
    if (dt_ms > 0U) {
      const float dt_s = ((float)dt_ms) * 0.001f;
      const float qdot1 = (q1 - plan->prev_joint_deg[0]) / dt_s;
      const float qdot2 = (q2 - plan->prev_joint_deg[1]) / dt_s;
      const float qdot3 = (q3 - plan->prev_joint_deg[2]) / dt_s;

      const long cmd1 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot1));
      const long cmd2 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot2));
      const long cmd3 = clamp_speed_cmd(joint_deg_s_to_speed_cmd(qdot3));

      robot_runtime_set_joint_speed(cmd1, cmd2, cmd3);
    }
  }

  plan->prev_joint_deg[0] = q1;
  plan->prev_joint_deg[1] = q2;
  plan->prev_joint_deg[2] = q3;
  plan->prev_joint_valid = true;
  plan->prev_tick_ms = now_ms;
  robot->current_pos = setpoint;
}

void motion_execute_stop_all(void)
{
  robot_runtime_stop_joint_speed();
}
