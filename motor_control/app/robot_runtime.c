#include "robot_runtime.h"

#include <string.h>

#include "app_motor_comm.h"
#include "hw_uart.h"

static mailbox_t *s_mailbox = NULL;
static io_motor_com_t *s_motor_com = NULL;

void robot_runtime_bind(mailbox_t *mailbox, io_motor_com_t *motor_com)
{
  s_mailbox = mailbox;
  s_motor_com = motor_com;
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

  Turn_const_speed(s_motor_com, ROBOT_MOTOR_1_ID, cmd1);
  Turn_const_speed(s_motor_com, ROBOT_MOTOR_2_ID, cmd2);
  Turn_const_speed(s_motor_com, ROBOT_MOTOR_3_ID, cmd3);
}

void robot_runtime_stop_joint_speed(void)
{
  robot_runtime_set_joint_speed(0, 0, 0);
}
