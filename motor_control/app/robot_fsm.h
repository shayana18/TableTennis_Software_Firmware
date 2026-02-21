#pragma once

#include "io_motor_com.h"
#include "mailbox.h"
#include "robot.h"


void delta_fsm_init(robot_t *robot);
void delta_fsm(robot_t *robot);
