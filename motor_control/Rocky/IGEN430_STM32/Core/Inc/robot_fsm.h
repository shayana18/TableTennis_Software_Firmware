/*
 * fsm.h
 *
 *  Created on: Feb 5, 2026
 *      Author: rocky
 */

#ifndef INC_ROBOT_FSM_H_
#define INC_ROBOT_FSM_H_

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "robot.h"


void delta_fsm_init(robot_t *robot);
void delta_fsm(robot_t *robot);


#endif /* INC_ROBOT_FSM_H_ */
