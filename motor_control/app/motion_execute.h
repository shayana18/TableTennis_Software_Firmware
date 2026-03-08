#pragma once

#include <stdbool.h>

#include "robot.h"

void motion_execute_reset_scheduler(void);
void motion_execute_on_timer_tick(void);
bool motion_execute_consume_tick_due(void);

// Check if joint angles are within limits. Returns true if all valid, false if any out of range.
bool motion_execute_safety_check_joint_limits(float q1_deg, float q2_deg, float q3_deg);

void motion_execute_make_home_target(robot_t *robot);
void motion_execute_prepare_strike(robot_t *robot);

void motion_execute_plan(robot_t *robot);
void motion_execute_start(robot_t *robot);
void motion_execute_tick(robot_t *robot);
void motion_execute_stop_all(void);
