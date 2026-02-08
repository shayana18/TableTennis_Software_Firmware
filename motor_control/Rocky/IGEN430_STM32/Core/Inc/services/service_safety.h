/*
 * service_safety.h
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#pragma once
#ifndef INC_SERVICES_SERVICE_SAFETY_H_
#define INC_SERVICES_SERVICE_SAFETY_H_

#include "robot.h"
bool safety_check_joint_limits(void);
bool safety_check_target(vec3 pos);


#endif /* INC_SERVICES_SERVICE_SAFETY_H_ */
