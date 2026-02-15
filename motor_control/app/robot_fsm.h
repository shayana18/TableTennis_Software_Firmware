#pragma once

#include "io_motor_com.h"
#include "mailbox.h"
#include "robot.h"

void delta_fsm_bind_io(mailbox_t *mailbox, io_motor_com_t *motor_com);
void delta_fsm_on_timer_tick(void);

void delta_fsm_init(robot_t *robot);
void delta_fsm(robot_t *robot);
