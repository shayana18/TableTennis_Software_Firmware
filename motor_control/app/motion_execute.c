#include "motion_execute.h"

#include <math.h>

#include "main.h"
#include "robot_runtime.h"

static volatile uint8_t s_motion_tick_pending = 0U;
static uint32_t s_tick_ms_accum = 0U;


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
  // Phase timings are referenced from motion start (t=0):
  // [0, t2): accel, [t2, t3): cruise, [t3, t3+t_dec): decel.
  // Any additional wait is represented by plan->t1 and applied after arrival.
  const float t_acc = plan->t2;
  const float t_cruise = plan->t3 - plan->t2;
  const float t_dec = ((plan->T - plan->t1) > plan->t3) ? ((plan->T - plan->t1) - plan->t3) : 0.0f;
  const float a = (plan->max_cart_acc > 1e-6f) ? plan->max_cart_acc : MAX_CART_ACC;
  const float d = (plan->max_cart_dec > 1e-6f) ? plan->max_cart_dec : MAX_CART_DEC;
  const float t_move_end = plan->t3 + t_dec;

  if (t_s <= 0.0f) {
    return 0.0f;
  }

  if (t_s < plan->t2) {
    const float dt = t_s;
    return 0.5f * a * dt * dt;
  }

  const float v_peak = a * t_acc;
  const float d_acc = 0.5f * a * t_acc * t_acc;

  if (t_s < plan->t3) {
    const float dt = t_s - plan->t2;
    return d_acc + v_peak * dt;
  }

  if (t_s < t_move_end) {
    const float dt = t_s - plan->t3;
    const float d_cruise = v_peak * t_cruise;
    return d_acc + d_cruise + v_peak * dt - 0.5f * d * dt * dt;
  }

  return plan->D;
}

static float clamp01(float value)
{
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 1.0f) {
    return 1.0f;
  }
  return value;
}

static float strike_tau_start(float tau_window_s)
{
  return tau_window_s * STRIKE_TAU_POS_SCALE;
}

static float strike_tau_end(float tau_window_s)
{
  return -tau_window_s * STRIKE_TAU_NEG_SCALE;
}

static float strike_tau_from_progress(float tau_window_s, float progress)
{
  const float tau_start = strike_tau_start(tau_window_s);
  const float tau_end = strike_tau_end(tau_window_s);
  return tau_start + ((tau_end - tau_start) * progress);
}

static vec3 strike_ball_pos_at_tau(vec3 intercept_pos, vec3 intercept_vel, float tau_s)
{
  vec3 out = {
      intercept_pos.x + (intercept_vel.x * tau_s),
      intercept_pos.y + (intercept_vel.y * tau_s),
      intercept_pos.z + (intercept_vel.z * tau_s) - (0.5f * GRAVITY * tau_s * tau_s),
  };
  return out;
}

static vec3 strike_ee_pos_from_ball(vec3 ball_pos, float paddle_yaw_rad)
{
  vec3 ee = {
      ball_pos.x - (PADDLE_X_OFFSET * cosf(paddle_yaw_rad) - PADDLE_Y_OFFSET * sinf(paddle_yaw_rad)),
      ball_pos.y - (PADDLE_X_OFFSET * sinf(paddle_yaw_rad) + PADDLE_Y_OFFSET * cosf(paddle_yaw_rad)),
      ball_pos.z - PADDLE_OFFSET_Z,
  };
  return ee;
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

  char msg[100];

  // Check if all joint angles are within the configured limits.
  if (q1_deg < MIN_JOINT_ANGLE_LIMIT || q1_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q1_deg_rounded = (long)lroundf(q1_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 1 Angle: %ld deg\r\n", q1_deg_rounded);
    robot_runtime_send_status(msg);
    long q1_deg_rounded = (long)lroundf(q1_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 1 Angle: %ld deg\r\n", q1_deg_rounded);
    robot_runtime_send_status(msg);
    return false;
  }
  if (q2_deg < MIN_JOINT_ANGLE_LIMIT || q2_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q2_deg_rounded = (long)lroundf(q2_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 2 Angle: %ld deg\r\n", q2_deg_rounded);
    robot_runtime_send_status(msg);
    long q2_deg_rounded = (long)lroundf(q2_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 2 Angle: %ld deg\r\n", q2_deg_rounded);
    robot_runtime_send_status(msg);
    return false;
  }
  if (q3_deg < MIN_JOINT_ANGLE_LIMIT || q3_deg > MAX_JOINT_ANGLE_LIMIT) {
    long q3_deg_rounded = (long)lroundf(q3_deg);
    snprintf(msg, sizeof(msg), "Invalid Joint 3 Angle: %ld deg\r\n", q3_deg_rounded);
    robot_runtime_send_status(msg);
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
  // Keep paddle axis aligned to its zero reference whenever we send the robot home.
  robot_runtime_set_paddle_abs_deg(90.0);
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
  const vec3 interception_target = robot->current_target.pos;
  const vec3 ball_vel = robot->current_target.vel;
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

  // Always command max cart strike speed.
  const float paddle_speed = MAX_STRIKE_VEL;

  if (yaw_norm < 0.0f) {
    motion_abort(robot, "Paddle yaw is negative?? \r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }
  float paddle_yaw = yaw_norm;

  // Pick one face deterministically at the 90 deg boundary (avoid the old
  // special-case that pointed the axis to 90 deg, which mismatched both sides).
  if (yaw_norm <= HALF_PI_F) {
    paddle_yaw += HALF_PI_F;
  } else {
    paddle_yaw -= HALF_PI_F;
  }
  paddle_yaw += (STRIKE_YAW_BIAS_DEG * DTR);
  while (paddle_yaw >= PI_F) {
    paddle_yaw -= PI_F;
  }
  while (paddle_yaw < 0.0f) {
    paddle_yaw += PI_F;
  }

  // Offset the EE using the commanded paddle yaw frame (mechanical reference).
  const float paddle_yaw_rad = paddle_yaw;
  const float paddle_yaw_deg = paddle_yaw * (180.0f / PI_F);
  const vec3 ee_interception_target = strike_ee_pos_from_ball(interception_target, paddle_yaw_rad);

  // EE pose that places paddle center at interception point (no strike sweep).
  new_interception_target = ee_interception_target;
  strike_plan->yaw_angle_deg = paddle_yaw_deg;
  robot->current_move_plan.yaw_angle_deg = paddle_yaw_deg;

  if (!robot_EE_in_workspace(new_interception_target)) {
    motion_abort(robot, "INTERCEPT TARGET OUT OF WORKSPACE\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }

  // Keep sweep direction in the return plane, independent of incoming ball speed.
  const vec3 strike_dir = {nx, ny, 0.0f};


  const float nominal_stop_dist = (paddle_speed * paddle_speed) / (2.0f * MAX_CART_DEC);
  const float interception_target_offset =
      (nominal_stop_dist + STRIKE_BUFFER_DIST + STRIKE_SWEEP_EXTRA_DIST) * STRIKE_SWEEP_DIST_GAIN;
  const float windup_offset_nominal = interception_target_offset * STRIKE_WINDUP_DIST_GAIN;
  const float followthrough_offset_nominal = interception_target_offset * STRIKE_FOLLOWTHROUGH_DIST_GAIN;
  const bool use_ballistic_path = false;

  // Use a deterministic straight-line sweep with larger windup to build speed before contact.
  bool strike_targets_valid = false;
  float windup_offset_selected = 0.0f;
  float followthrough_offset_selected = 0.0f;
  float sweep_scale = 1.0f;
  for (int attempt = 0; attempt < 15 && !strike_targets_valid; attempt++) {
    const float windup_offset = windup_offset_nominal * sweep_scale;
    const float followthrough_offset = followthrough_offset_nominal * sweep_scale;

    new_interception_target.x = ee_interception_target.x - (strike_dir.x * windup_offset);
    new_interception_target.y = ee_interception_target.y - (strike_dir.y * windup_offset);
    new_interception_target.z = ee_interception_target.z;

    strike_finish_target.x = ee_interception_target.x + (strike_dir.x * followthrough_offset);
    strike_finish_target.y = ee_interception_target.y + (strike_dir.y * followthrough_offset);
    strike_finish_target.z = ee_interception_target.z;

    strike_targets_valid = robot_EE_in_workspace(new_interception_target) &&
                           robot_EE_in_workspace(strike_finish_target);
    if (strike_targets_valid) {
      windup_offset_selected = windup_offset;
      followthrough_offset_selected = followthrough_offset;
    }
    sweep_scale *= 0.96f;
  }

  if (!strike_targets_valid) {
    motion_abort(robot, "STRIKE SWEEP OUT OF WORKSPACE\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }
  float contact_progress = 0.5f;
  const float strike_span = windup_offset_selected + followthrough_offset_selected;
  if (strike_span > 1e-6f) {
    contact_progress = clamp01(windup_offset_selected / strike_span);
  }

  robot_runtime_send_status("STRIKE MODE: POWER LINEAR SWEEP MAX SPEED\r\n");

  float line_dx;
  float line_dy;
  float line_dz;
  const float line_dist = robot_calc_dist(new_interception_target, strike_finish_target, &line_dx, &line_dy, &line_dz);
  const float D = line_dist;

  if (!(D > 1.0f) || !(line_dist > 1.0f)) {
    motion_abort(robot, "STRIKE SWEEP TOO SHORT\r\n");
    robot->state = STATE_IDLE;
    set_idle(robot);
    robot_runtime_send_status("STATE: IDLE\r\n");
    return;
  }

  // Update the interception target
  robot->current_target.pos = new_interception_target;

  // Update the strike target
  strike_plan->start_pos = new_interception_target;
  strike_plan->target_pos = strike_finish_target;
  strike_plan->D = D;
  strike_plan->yaw_angle_deg = paddle_yaw_deg;
  strike_plan->max_cart_vel = paddle_speed;
  strike_plan->max_cart_acc = MAX_CART_ACC;
  strike_plan->max_cart_dec = MAX_CART_DEC;
  strike_plan->ballistic_track = use_ballistic_path;
  strike_plan->ballistic_intercept_pos = interception_target;
  strike_plan->ballistic_intercept_vel = ball_vel;
  strike_plan->ballistic_tau_window_s = 0.0f;
  strike_plan->strike_contact_progress = contact_progress;

  strike_plan->dir.x = line_dx / line_dist;
  strike_plan->dir.y = line_dy / line_dist;
  strike_plan->dir.z = line_dz / line_dist;

  // Strike Planning
  float t_acc;
  float t_dec;
  float t_cruise;
  const float a = (strike_plan->max_cart_acc > 1e-6f) ? strike_plan->max_cart_acc : MAX_CART_ACC;
  const float d = (strike_plan->max_cart_dec > 1e-6f) ? strike_plan->max_cart_dec : MAX_CART_DEC;
  const float d_acc_at_vmax =
      (strike_plan->max_cart_vel * strike_plan->max_cart_vel) / (2.0f * a);
  const float d_dec_at_vmax =
      (strike_plan->max_cart_vel * strike_plan->max_cart_vel) / (2.0f * d);
  if (strike_plan->D <= (d_acc_at_vmax + d_dec_at_vmax)) {
    const float v_peak = sqrtf((2.0f * strike_plan->D * a * d) / (a + d));
    t_acc = v_peak / a;
    t_dec = v_peak / d;
    t_cruise = 0.0f;
    strike_plan->max_cart_vel = v_peak;
  } else {
    t_acc = strike_plan->max_cart_vel / a;
    t_dec = strike_plan->max_cart_vel / d;
    t_cruise = (strike_plan->D - d_acc_at_vmax - d_dec_at_vmax) / strike_plan->max_cart_vel;
  }

  strike_plan->t1 = 0.0f;
  strike_plan->t2 = t_acc;
  strike_plan->t3 = strike_plan->t2 + t_cruise;
  strike_plan->T = strike_plan->t3 + t_dec;

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
    strike_time_buffer = robot->strike_move_plan.T * robot->strike_move_plan.strike_contact_progress;
    if (robot->strike_move_plan.T > 0.0f) {
      strike_time_buffer += STRIKE_CONTACT_LEAD_TIME;
    }

    if (!robot->strike_move_plan.active) {
      robot_runtime_send_status("STRIKE PLAN FAILED\r\n");
      return;
    }
  }

  const vec3 target = robot->current_target.pos;

  if (!robot_EE_in_workspace(target)) {
    motion_abort(robot, "PATH_ABORT: EE OUT OF WORKSPACE\r\n");
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
  plan->max_cart_acc = (robot->current_target.type == TARGET_HOME) ? HOME_CART_ACC : MAX_CART_ACC;
  plan->max_cart_dec = (robot->current_target.type == TARGET_HOME) ? HOME_CART_DEC : MAX_CART_DEC;
  plan->max_cart_vel = MAX_CART_VEL;


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
  float t_dec;
  float t_cruise;
  const float a = (plan->max_cart_acc > 1e-6f) ? plan->max_cart_acc : MAX_CART_ACC;
  const float d = (plan->max_cart_dec > 1e-6f) ? plan->max_cart_dec : MAX_CART_DEC;
  const float d_acc_at_vmax = (plan->max_cart_vel * plan->max_cart_vel) / (2.0f * a);
  const float d_dec_at_vmax = (plan->max_cart_vel * plan->max_cart_vel) / (2.0f * d);
  if (D <= (d_acc_at_vmax + d_dec_at_vmax)) {
    const float v_peak = sqrtf((2.0f * D * a * d) / (a + d));
    t_acc = v_peak / a;
    t_dec = v_peak / d;
    t_cruise = 0.0f;
    plan->max_cart_vel = v_peak;
  } else {
    t_acc = plan->max_cart_vel / a;
    t_dec = plan->max_cart_vel / d;
    t_cruise = (D - d_acc_at_vmax - d_dec_at_vmax) / plan->max_cart_vel;
  }

  const float t_move = t_acc + t_cruise + t_dec; // Total move time
  const float t_processing_time_buffer = ((float)(HAL_GetTick() - robot->current_target.received_time)) * 0.001f;


  float t_extra = robot->current_target.t_arrival_s - t_move - strike_time_buffer - t_processing_time_buffer - BUFFER_TIME;

  if (t_extra < 0.0f) {
    t_extra = 0.0f;
    robot_runtime_send_status("WARN: Robot will be late\r\n");
    robot_runtime_send_status("WARN: Robot will be late\r\n");
  }

  // Move immediately, then hold at target for t_extra.
  plan->t1 = t_extra;  // post-arrival hold duration
  plan->t2 = t_acc;
  plan->t3 = plan->t2 + t_cruise;
  plan->T = plan->t3 + t_dec + plan->t1;
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
    set_idle(robot);
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

  vec3 setpoint;
  if (robot->current_target.type == TARGET_STRIKE && plan->ballistic_track) {
    const float progress = clamp01((plan->D > 1e-6f) ? (s / plan->D) : 1.0f);
    const float tau = strike_tau_from_progress(plan->ballistic_tau_window_s, progress);
    const vec3 ball_pos = strike_ball_pos_at_tau(plan->ballistic_intercept_pos, plan->ballistic_intercept_vel, tau);
    setpoint = strike_ee_pos_from_ball(ball_pos, plan->yaw_angle_deg * DTR);
  } else {
    setpoint = (vec3){
        plan->start_pos.x + plan->dir.x * s,
        plan->start_pos.y + plan->dir.y * s,
        plan->start_pos.z + plan->dir.z * s,
    };
  }

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
  const bool joints_ok = motion_execute_safety_check_joint_limits(q1_ik, q2_ik, q3_ik);
  if (!joints_ok && !(robot->current_target.type == TARGET_HOME)) {
    motion_abort(robot, "PATH_ABORT: ROBOT JOINT LIMITS EXCEEDED\r\n");
    set_idle(robot);
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
