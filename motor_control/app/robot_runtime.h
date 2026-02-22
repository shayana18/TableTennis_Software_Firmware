#pragma once

#include <stdbool.h>

#include "io_motor_com.h"
#include "mailbox.h"
#include "robot.h"

void robot_runtime_bind(mailbox_t *mailbox, io_motor_com_t *motor_com);

void robot_runtime_send_status(const char *msg);
bool robot_runtime_pop_target(robot_target_t *out_target);

void robot_runtime_set_joint_speed(long cmd1, long cmd2, long cmd3);
void robot_runtime_stop_joint_speed(void);
void robot_runtime_set_joint_position_abs_ticks(long q1_tick, long q2_tick, long q3_tick);
void robot_runtime_set_joint_position_abs_deg(float q1_deg, float q2_deg, float q3_deg);

bool robot_runtime_get_joint_ticks(long *q1_tick, long *q2_tick, long *q3_tick);

void robot_runtime_scan_motor_ids(char first_id, char last_id);
