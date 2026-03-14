#include "motion_execute.h"

#include <math.h>

#include "main.h"
#include "robot_runtime.h"

static volatile uint8_t s_motion_tick_pending = 0U;
static uint32_t s_tick_ms_accum = 0U;


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

  robot->strike_move_plan.active = false;
  robot->current_move_plan.active = false;
  robot->flag_ready_to_move = false;
  robot->flag_path_abort = true;

  // setting velocity to 0 makes robot jerk! Revisit
  //robot_runtime_stop_joint_speed();
  if (reason != NULL) {
    robot_runtime_send_status(reason);
  }
}

static void motion_finish(robot_t *robot, move_plan *plan)
{
  // Legacy velocity-mode stop behavior:
  // robot_runtime_stop_joint_speed();

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
  char msg[100];

  // Check if all joint angles are within the configured limits.
  if (q1_deg < MIN_JOINT_ANGLE_LIMIT || q1_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q1_deg_rounded = (long)lroundf(q1_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 1 Angle: %ld deg\r\n", q1_deg_rounded);
    robot_runtime_send_status(msg);
    return false;
  }
  if (q2_deg < MIN_JOINT_ANGLE_LIMIT || q2_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q2_deg_rounded = (long)lroundf(q2_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 2 Angle: %ld deg\r\n", q2_deg_rounded);
    robot_runtime_send_status(msg);
    return false;
  }
  if (q3_deg < MIN_JOINT_ANGLE_LIMIT || q3_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q3_deg_rounded = (long)lroundf(q3_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 3 Angle: %ld deg\r\n", q3_deg_rounded);
    robot_runtime_send_status(msg);
    return false;
  }
  return true;
}

void motion_execute_make_home_target(robot_t *robot)
{
  robot->current_target.type = TARGET_HOME;
  robot->current_target.pos = home;
  robot->current_target.t_arrival_s = HOME_TIME;
  robot->current_target.received_time = HAL_GetTick();
}

void motion_execute_prepare_strike(robot_t *robot)
{

  move_plan *plan = &robot->strike_move_plan;

  const uint32_t now_ms = HAL_GetTick();
  plan->t_start_ms = now_ms;
  plan->prev_joint_deg[0] = robot->current_move_plan.prev_joint_deg[0];
  plan->prev_joint_deg[1] = robot->current_move_plan.prev_joint_deg[1];
  plan->prev_joint_deg[2] = robot->current_move_plan.prev_joint_deg[2];
  plan->prev_joint_valid = robot->current_move_plan.prev_joint_valid;

  robot->current_target.pos = plan->target_pos;
  robot->current_target.type = TARGET_STRIKE;

  robot->flag_path_done = false;
  robot->flag_path_abort = false;

}

void motion_execute_plan_strike(robot_t *robot) {
  // Return the target offset we want to move to and the 4th axis yaw_norm angle
  vec3 new_interception_target;
  vec3 interception_target = robot->current_target.pos;
  vec3 ball_vel = robot->current_target.vel; 
  vec3 strike_finish_target;
  move_plan *strike_plan = &robot->strike_move_plan;

  const float disc = (ball_vel.z * ball_vel.z) + (2.0f * GRAVITY * (interception_target.z - STRIKE_TARGET_Z));
  const float det = sqrtf(disc);
  const float ball_return_travel_time = (-ball_vel.z - det) / (-GRAVITY);  // later root

  if (!(ball_return_travel_time > 0.0f)) {
    motion_abort(robot, "BALL RETURN TIME NEGATIVE or NaN\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }

  float vout_x = ((STRIKE_TARGET_X - interception_target.x) / ball_return_travel_time);
  float vout_y = ((STRIKE_TARGET_Y - interception_target.y) / ball_return_travel_time);

  float dvx = vout_x - ball_vel.x;
  float dvy = vout_y - ball_vel.y;

  float yaw_norm = atan2f(dvy, dvx);

  float nx = cosf(yaw_norm);
  float ny = sinf(yaw_norm);

  float vin_n  = ball_vel.x * nx + ball_vel.y * ny;
  float vout_n = vout_x* nx + vout_y* ny;
  float vp_n   = (vout_n + RESTITUTION * vin_n) / (1.0f + RESTITUTION);

  float paddle_speed = fabsf(vp_n);
  
  if (paddle_speed < 1.0f) {

    // Update the strike target
    strike_plan->start_pos = interception_target;
    strike_plan->target_pos = interception_target;
    strike_plan->D = 0.0f;

    strike_plan->dir.x = 0.0;
    strike_plan->dir.y = 0.0;
    strike_plan->dir.z = 0.0;

    strike_plan->t1 = 0.0f;
    strike_plan->t2 = 0.0f;
    strike_plan->t3 = 0.0f;
    strike_plan->T = 0.0f;
    strike_plan->active = true;
    robot->current_move_plan.yaw_angle_deg = 0.0;

    return;
  }


  if (paddle_speed > MAX_STRIKE_VEL) {
    paddle_speed = MAX_STRIKE_VEL;
  }

  float dir_sign = (vp_n >= 0.0f) ? 1.0f : -1.0f;
  float dir_x = dir_sign * nx;
  float dir_y = dir_sign * ny;


  float interception_target_offset = paddle_speed * paddle_speed / (2.0f * MAX_CART_ACC) + STRIKE_BUFFER_DIST;

  if (yaw_norm < 0.0f) {
    motion_abort(robot, "Paddle yaw is negative?? \r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }
  float paddle_yaw = yaw_norm; 

  if (yaw_norm < (HALF_PI_F)) {
    paddle_yaw += HALF_PI_F;  
  } else if (yaw_norm > (HALF_PI_F)) {
    paddle_yaw -= HALF_PI_F;
  } else {
    paddle_yaw = 0.0f; 
  }

  float x_offset = PADDLE_ARM_OFFSET * nx;
  float y_offset = PADDLE_ARM_OFFSET * ny;

  new_interception_target.x = (interception_target.x - x_offset) - (dir_x * interception_target_offset);
  new_interception_target.y = (interception_target.y - y_offset) - (dir_y * interception_target_offset); 
  new_interception_target.z = interception_target.z - PADDLE_OFFSET_Z;

  strike_finish_target.x = interception_target.x - x_offset + (dir_x * interception_target_offset);
  strike_finish_target.y = interception_target.y - y_offset + (dir_y * interception_target_offset); 
  strike_finish_target.z = interception_target.z - PADDLE_OFFSET_Z;

  if (!robot_EE_in_workspace(strike_finish_target)) {
    motion_abort(robot, "STRIKE TARGET OUT OF WORKSPACE\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }

  float dx, dy, dz;
  const float D = robot_calc_dist(new_interception_target, strike_finish_target, &dx, &dy, &dz);

  // Update the interception target
  robot->current_target.pos = new_interception_target;
  robot->current_move_plan.yaw_angle_deg = paddle_yaw * (180.0f / PI_F);

  // Update the strike target
  strike_plan->start_pos = new_interception_target;
  strike_plan->target_pos = strike_finish_target;
  strike_plan->D = D;

  strike_plan->max_cart_vel = paddle_speed;

  strike_plan->dir.x = dir_x;
  strike_plan->dir.y = dir_y;
  strike_plan->dir.z = 0.0f;

  // Strike Planning
  float t_acc = strike_plan->max_cart_vel / MAX_CART_ACC;
  float ramp_dist = 0.5f * MAX_CART_ACC * (t_acc * t_acc);
  float t_cruise = (strike_plan->D - 2.0f * ramp_dist) / strike_plan->max_cart_vel;

  strike_plan->t1 = 0.0f;
  strike_plan->t2 = strike_plan->t1 + t_acc;
  strike_plan->t3 = strike_plan->t2 + t_cruise;
  strike_plan->T = strike_plan->t3 + t_acc;

  strike_plan->active = true;
  

}

void motion_execute_plan(robot_t *robot)
{
  move_plan *plan = &robot->current_move_plan;
  const uint32_t now_ms = HAL_GetTick();
  float strike_time_buffer = 0.0f;

  // Do not plan offset if target is Home or test
  if (robot->current_target.type == TARGET_INTERCEPT) {
    if (!robot_target_in_workspace(robot->current_target.pos)) {
      motion_abort(robot, "PATH_ABORT: TARGET OUT OF WORKSPACE\r\n");
      robot->state = STATE_IDLE;
      set_idle(robot);
      robot_runtime_send_status("STATE: IDLE\r\n");
      return;
    }
    motion_execute_plan_strike(robot);
    strike_time_buffer = robot->strike_move_plan.T / 2.0f;

    if (!robot->strike_move_plan.active) {
      robot_runtime_send_status("STRIKE PLAN FAILED\r\n");
      return;
    }
  }

  const vec3 target = robot->current_target.pos;

  if (!robot_EE_in_workspace(target)) {
    motion_abort(robot, "PATH_ABORT: TARGET OUT OF WORKSPACE\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  } 

  // Use a single joint sample to seed both start pose and unwrap reference.
  float q1_now, q2_now, q3_now;
  if (!robot_get_joint_angles(&q1_now, &q2_now, &q3_now)) {
    plan->prev_joint_valid = false;
    robot_runtime_send_status("ERR: prev q unset\r\n");
    return;
  }

  const vec3 start = FK(q1_now, q2_now, q3_now);
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

  // Seed previous joint state from the same sample used for start pose.
  plan->prev_joint_deg[0] = q1_now;
  plan->prev_joint_deg[1] = q2_now;
  plan->prev_joint_deg[2] = q3_now;
  plan->prev_joint_valid = true;

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

  const float t_move = (2.0f * t_acc) + t_cruise; // Total move time
  const float t_processing_time_buffer = ((float)(HAL_GetTick() - robot->current_target.received_time)) * 0.001f;


  float t_extra = robot->current_target.t_arrival_s - t_move - strike_time_buffer - t_processing_time_buffer - BUFFER_TIME;

  if (t_extra < 0.0f) {
    t_extra = 0.0f;
    robot_runtime_send_status("WARN: Robot will be late\r\n");
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
  move_plan *plan;

  if (robot->current_target.type == TARGET_STRIKE) {
    plan = &robot->strike_move_plan;

  }else {
    plan = &robot->current_move_plan;
  }

  if (!plan->active || !plan->prev_joint_valid) {
    robot->flag_ready_to_move = false;
    robot_runtime_send_status("PLAN_ABORT: PLANNING FAILED\r\n");
    return;
  }

  if (robot->current_target.type == TARGET_INTERCEPT) {
    robot_runtime_set_paddle_abs_deg(plan->yaw_angle_deg);
  }
  robot->flag_ready_to_move = true;
}

void motion_execute_tick(robot_t *robot)
{
  move_plan *plan;
  if (robot->current_target.type == TARGET_STRIKE) {
    plan = &robot->strike_move_plan;

  }else {
    plan = &robot->current_move_plan;
  }

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

  const bool joints_ok = motion_execute_safety_check_joint_limits(q1_ik, q2_ik, q3_ik);
  if (!joints_ok && !(robot->current_target.type == TARGET_HOME)) {
    motion_abort(robot, "PATH_ABORT: ROBOT JOINT LIMITS EXCEEDED\r\n");
    return;
  }

  robot_runtime_set_joint_position_abs_deg(q1_ik, q2_ik, q3_ik);

  plan->prev_joint_deg[0] = q1_ik;
  plan->prev_joint_deg[1] = q2_ik;
  plan->prev_joint_deg[2] = q3_ik;
  plan->prev_joint_valid = true;
  plan->prev_tick_ms = now_ms;
  robot->current_pos = setpoint;

  if (t_s >= plan->T) {
    motion_finish(robot, plan);
    return;
  }
}



void motion_execute_stop_all(void)
{
  // Legacy velocity-mode behavior:
  // robot_runtime_stop_joint_speed();

  // In position-control mode we avoid overriding the final absolute target.
}
