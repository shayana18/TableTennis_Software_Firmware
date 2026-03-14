#include "robot_runtime.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "app_motor_comm.h"
#include "hw_uart.h"

static mailbox_t *s_mailbox = NULL;
static io_motor_com_t *s_motor_com = NULL;
static bool s_speed_cmd_valid = false;
static long s_speed_cmd_last_1 = 0;
static long s_speed_cmd_last_2 = 0;
static long s_speed_cmd_last_3 = 0;

static long robot_runtime_joint_deg_to_motor_tick(float q_deg);

void robot_runtime_bind(mailbox_t *mailbox, io_motor_com_t *motor_com)
{
  s_mailbox = mailbox;
  s_motor_com = motor_com;
  s_speed_cmd_valid = false;
}

void robot_runtime_send_status(const char *msg)
{
  if (msg == NULL) {
    return;
  }
  hw_laptop_tx((const uint8_t *)msg, (uint16_t)strlen(msg), 20);
}

bool robot_runtime_pop_target(robot_target_t *out_target)
{
  target_t mailbox_target;

  if (s_mailbox == NULL || out_target == NULL) {
    return false;
  }

  if (!mailbox_mail_received(s_mailbox, &mailbox_target)) {
    return false;
  }

  robot_set_target_from_mail(out_target, &mailbox_target);
  return true;
}

void robot_runtime_clear_mailbox()
{
  if (s_mailbox != NULL) {
    mailbox_clear(s_mailbox);
  }
}

void robot_runtime_set_joint_speed(long cmd1, long cmd2, long cmd3)
{
  if (s_motor_com == NULL) {
    return;
  }

  const long motor_cmd1 = (long)lroundf((float)cmd1 * ROBOT_JOINT_SIGN);
  const long motor_cmd2 = (long)lroundf((float)cmd2 * ROBOT_JOINT_SIGN);
  const long motor_cmd3 = (long)lroundf((float)cmd3 * ROBOT_JOINT_SIGN);

  if (!s_speed_cmd_valid || motor_cmd1 != s_speed_cmd_last_1) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_2_ID, motor_cmd1);
  }
  if (!s_speed_cmd_valid || motor_cmd2 != s_speed_cmd_last_2) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_3_ID, motor_cmd2);
  }
  if (!s_speed_cmd_valid || motor_cmd3 != s_speed_cmd_last_3) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_4_ID, motor_cmd3);
  }

  s_speed_cmd_last_1 = motor_cmd1;
  s_speed_cmd_last_2 = motor_cmd2;
  s_speed_cmd_last_3 = motor_cmd3;
  s_speed_cmd_valid = true;
}

void robot_runtime_stop_joint_speed(void)
{
  robot_runtime_set_joint_speed(0, 0, 0);
}

static long robot_runtime_joint_deg_to_motor_tick(float q_deg)
{
  const float ticks_per_joint_deg = ((float)PULSES_PER_REV * JOINT_GEAR_RATIO) / 360.0f;
  const float motor_tick = (q_deg - ENCODER_ZERO_TO_IK_OFFSET) * ticks_per_joint_deg * ROBOT_JOINT_SIGN;
  return (long)lroundf(motor_tick);
}

void robot_runtime_set_joint_position_abs_ticks(long q1_tick, long q2_tick, long q3_tick)
{
  if (s_motor_com == NULL) {
    return;
  }

  move_abs32(s_motor_com, ROBOT_MOTOR_2_ID, q1_tick);
  move_abs32(s_motor_com, ROBOT_MOTOR_3_ID, q2_tick);
  move_abs32(s_motor_com, ROBOT_MOTOR_4_ID, q3_tick);
}

void robot_runtime_set_joint_position_abs_deg(float q1_deg, float q2_deg, float q3_deg)
{
  const long q1_tick = robot_runtime_joint_deg_to_motor_tick(q1_deg);
  const long q2_tick = robot_runtime_joint_deg_to_motor_tick(q2_deg);
  const long q3_tick = robot_runtime_joint_deg_to_motor_tick(q3_deg);

  robot_runtime_set_joint_position_abs_ticks(q1_tick, q2_tick, q3_tick);
}

void robot_runtime_set_paddle_abs_deg(float paddle_yaw_deg) {

  if (s_motor_com == NULL) {
    robot_runtime_send_status("PADDLE CMD SKIPPED: motor comm not bound\r\n");
    return;
  }
  const float motor_tick = paddle_yaw_deg * ((float)PULSES_PER_REV) / 360.0f;
  long paddle_tick = (long)lroundf(motor_tick);
  long paddle_cdeg = (long)lroundf(paddle_yaw_deg * 100.0f);
  long paddle_pos_before_tick = 0;
  bool paddle_pos_before_valid = false;

  if (ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_1_ID)) {
    paddle_pos_before_tick = io_motor_com_get_motor_pos(s_motor_com);
    paddle_pos_before_valid = true;
  }

  long delta_tick = paddle_tick - paddle_pos_before_tick;
  long delta_cdeg = (long)lroundf(((float)delta_tick) * (36000.0f / (float)PULSES_PER_REV));

  char msg[128];
  if (paddle_pos_before_valid) {
    snprintf(msg, sizeof(msg),
             "PADDLE CMD: %ld cdeg tick=%ld from=%ld dtick=%ld dcdeg=%ld\r\n",
             paddle_cdeg, paddle_tick, paddle_pos_before_tick, delta_tick, delta_cdeg);
  } else {
    snprintf(msg, sizeof(msg), "PADDLE CMD: %ld cdeg tick=%ld (pre-read timeout)\r\n",
             paddle_cdeg, paddle_tick);
  }
  robot_runtime_send_status(msg);

  move_abs32(s_motor_com, ROBOT_MOTOR_1_ID, paddle_tick);

  if (ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_1_ID)) {
    long paddle_pos_tick = io_motor_com_get_motor_pos(s_motor_com);
    snprintf(msg, sizeof(msg), "PADDLE POS: tick=%ld, err=%ld\r\n", paddle_pos_tick, paddle_tick - paddle_pos_tick);
    robot_runtime_send_status(msg);
  } else {
    robot_runtime_send_status("ERR: paddle pos timeout\r\n");
  }
}


bool robot_runtime_get_joint_ticks(long *q1_tick, long *q2_tick, long *q3_tick)
{
  if (q1_tick == NULL || q2_tick == NULL || q3_tick == NULL) {
    return false;
  }

  if (s_motor_com == NULL) {
    return false;
  }

  // Request and read encoder positions from each motor driver
  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_2_ID)) {
    robot_runtime_send_status("ERR: m1 pos timeout\r\n");
    return false;
  }
  long p1 = io_motor_com_get_motor_pos(s_motor_com);

  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_3_ID)) {
    robot_runtime_send_status("ERR: m2 pos timeout\r\n");
    return false;
  }
  long p2 = io_motor_com_get_motor_pos(s_motor_com);

  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_4_ID)) {
    robot_runtime_send_status("ERR: m3 pos timeout\r\n");
    return false;
  }
  long p3 = io_motor_com_get_motor_pos(s_motor_com);

  *q1_tick = p1;
  *q2_tick = p2;
  *q3_tick = p3;

  return true;
}

void robot_runtime_scan_motor_ids(char first_id, char last_id)
{
  if (s_motor_com == NULL) {
    robot_runtime_send_status("MOTOR BUS SCAN skipped: motor comm not bound\r\n");
    return;
  }

  if (first_id > last_id) {
    char tmp = first_id;
    first_id = last_id;
    last_id = tmp;
  }

  char msg[128];
  snprintf(msg, sizeof(msg), "MOTOR BUS SCAN ids=%d..%d\r\n", first_id, last_id);
  robot_runtime_send_status(msg);

  for (char id = first_id; id <= last_id; id++) {
    if (ReadMotorPosition32Quiet(s_motor_com, id)) {
      long pos = io_motor_com_get_motor_pos(s_motor_com);
      snprintf(msg, sizeof(msg), "MOTOR BUS: id=%d OK pos=%ld\r\n", id, pos);
      robot_runtime_send_status(msg);
    } else {
      snprintf(msg, sizeof(msg), "MOTOR BUS: id=%d no response\r\n", id);
      robot_runtime_send_status(msg);
    }
  }
}
