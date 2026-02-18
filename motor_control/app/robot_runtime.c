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

void robot_runtime_set_joint_speed(long cmd1, long cmd2, long cmd3)
{
  if (s_motor_com == NULL) {
    return;
  }

  const long motor_cmd1 = (long)lroundf((float)cmd1 * ROBOT_JOINT_SIGN_1);
  const long motor_cmd2 = (long)lroundf((float)cmd2 * ROBOT_JOINT_SIGN_2);
  const long motor_cmd3 = (long)lroundf((float)cmd3 * ROBOT_JOINT_SIGN_3);

  if (!s_speed_cmd_valid || motor_cmd1 != s_speed_cmd_last_1) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_1_ID, motor_cmd1);
  }
  if (!s_speed_cmd_valid || motor_cmd2 != s_speed_cmd_last_2) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_2_ID, motor_cmd2);
  }
  if (!s_speed_cmd_valid || motor_cmd3 != s_speed_cmd_last_3) {
    Turn_const_speed(s_motor_com, ROBOT_MOTOR_3_ID, motor_cmd3);
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

bool robot_runtime_get_joint_angles(float *q1_deg, float *q2_deg, float *q3_deg)
{
  if (s_motor_com == NULL || q1_deg == NULL || q2_deg == NULL || q3_deg == NULL) {
    return false;
  }

  // Request and read encoder positions from each motor driver
  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_1_ID)) {
    robot_runtime_send_status("ERR: m1 pos timeout\r\n");
    return false;
  }
  long p1 = io_motor_com_get_motor_pos(s_motor_com) - (long)HOME_PULSE_OFFSET_M1;

  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_2_ID)) {
    robot_runtime_send_status("ERR: m2 pos timeout\r\n");
    return false;
  }
  long p2 = io_motor_com_get_motor_pos(s_motor_com) - (long)HOME_PULSE_OFFSET_M2;

  if (!ReadMotorPosition32(s_motor_com, ROBOT_MOTOR_3_ID)) {
    robot_runtime_send_status("ERR: m3 pos timeout\r\n");
    return false;
  }
  long p3 = io_motor_com_get_motor_pos(s_motor_com) - (long)HOME_PULSE_OFFSET_M3;

  float deg1 = (((float)p1 / PULSES_PER_REV) * 360.0f / JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN_1;
  float deg2 = (((float)p2 / PULSES_PER_REV) * 360.0f / JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN_2;
  float deg3 = (((float)p3 / PULSES_PER_REV) * 360.0f / JOINT_GEAR_RATIO) * ROBOT_JOINT_SIGN_3;

  // Keep absolute/continuous joint angle from encoder pulses.
  // Wrapping to [-180,180) can introduce artificial jumps near +/-180.

  *q1_deg = deg1;
  *q2_deg = deg2;
  *q3_deg = deg3;

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
