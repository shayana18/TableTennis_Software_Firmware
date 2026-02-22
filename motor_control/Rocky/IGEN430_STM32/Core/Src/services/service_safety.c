/*
 * service_safety.c
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#include "services/service_safety.h"

bool safety_check_joint_limits() {

	// READ MOTOR ANGLES!!!

	for (int i = 0; i < 3; i++) {
		float motor_angle = 0.0f;  // TODO: Read from motor encoders
		if (!(motor_angle <= MAX_JOINT_ANGLE_LIMIT && motor_angle >= MIN_JOINT_ANGLE_LIMIT)) {
			return false;
		}
	}
	return true;
}


bool safety_check_target(vec3 pos) {

	if (pos.x < LIMIT_NEG_X || pos.x > LIMIT_POS_X) {
		return false;
	}
	if (pos.y < LIMIT_NEG_Y || pos.y > LIMIT_POS_Y) {
		return false;
	}
	if (pos.z < LIMIT_NEG_Z || pos.z > LIMIT_POS_Z) {
		return false;
	}

	return true;

}
