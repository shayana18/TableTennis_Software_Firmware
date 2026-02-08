/*
 * robot.c
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#pragma once
#include "robot.h"
#include "services/service_comms.h"

//
#define PI 3.14159265358979323846
#define DTR PI / 180
#define SQRT3 1.7320508075688772
#define TAN30 1.0f / SQRT3
#define SIN30 0.5
#define TAN60 SQRT3
#define SIN120 0.8660254037844386
#define COS120 -0.5


vec3 robot_get_current_pos() {

	//

	float q1;
	float q2;
	float q3;


	comms_send_status("Encoder values read and returned\n");
	return FK(q1, q2, q3);
}


// DELTA ROBOT FORWARD KINEMATICS
vec3 FK(float motor_q1, float motor_q2, float motor_q3){


	float f = BASE_RADIUS * 2.0f;
	float e = EE_RADIUS * 2.0f;
	float rf = UPPER_ARM_LENGTH;
	float re = LOWER_ARM_LENGTH;

	// ---- convert to radians ----
	float t1 = motor_q1 * DTR;
	float t2 = motor_q2 * DTR;
	float t3 = motor_q3 * DTR;

	// ---- geometry helper ----
	float t = (f - e) * TAN30 * 0.5f;

	// ---- calculate the three arm joint positions ----
	float y1 = -(t + rf * cosf(t1));
	float z1 = -rf * sinf(t1);

	float y2 =  (t + rf * cosf(t2)) * SIN30;
	float x2 =  y2 * TAN60;
	float z2 = -rf * sinf(t2);

	float y3 =  (t + rf * cosf(t3)) * SIN30;
	float x3 = -y3 * TAN60;
	float z3 = -rf * sinf(t3);

	// ---- determinant ----
	float dnm = (y2 - y1) * x3 - (y3 - y1) * x2;

	// Protect against division by ~0 (singular configuration / numerical issue)
	if (fabsf(dnm) < 1e-9f) {
		return (vec3){0.0f, 0.0f, 0.0f};  // Singular, return zero position
	}

	float w1 = y1 * y1 + z1 * z1;
	float w2 = x2 * x2 + y2 * y2 + z2 * z2;
	float w3 = x3 * x3 + y3 * y3 + z3 * z3;

	// x = (a1*z + b1)/dnm
	float a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1);
	float b1 = -0.5f * ((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1));

	// y = (a2*z + b2)/dnm
	float a2 = -(z2 - z1) * x3 + (z3 - z1) * x2;
	float b2 =  0.5f * ((w2 - w1) * x3 - (w3 - w1) * x2);

	// a*z^2 + b*z + c = 0
	float A = a1 * a1 + a2 * a2 + dnm * dnm;
	float B = 2.0f * (a1 * b1 + a2 * b2 + dnm * dnm * z1);
	float C = (b1 * b1 + b2 * b2 + dnm * dnm * (z1 * z1 - re * re));

	float D = B * B - 4.0f * A * C;

	// Numerical robustness: allow tiny negative due to float roundoff
	if (D < 0.0f) {
		if (D > -1e-6f) D = 0.0f;
		else return (vec3){0.0f, 0.0f, 0.0f};  // No real solution
	}

	float sqrtD = sqrtf(D);

	// Choose the physically correct root for your coordinate convention.
	// This matches your original code (uses -(B + sqrtD)/(2A)).
	float z = -0.5f * (B + sqrtD) / A;
	float x = (a1 * z + b1) / dnm;
	float y = (a2 * z + b2) / dnm;

	return (vec3){x, y, z};

}

// Helper function for IK - calculates motor angle for YZ plane
// Returns 0 if successful, -1 if unreachable
static int calc_angleYZ(float x0, float y0, float z0, float *theta) {

    // Match your Python guard: if |z0| < 1 mm treat as unreachable
    if (fabsf(z0) < 1.0f) {
        return -1;
    }

    // Upper and lower joint locations in YZ plane
    float y1 = -BASE_RADIUS;
    float y0p = y0 - EE_RADIUS;

    // a and b from your derivation
    float a = (x0*x0 + y0p*y0p + z0*z0 + UPPER_ARM_LENGTH*UPPER_ARM_LENGTH - LOWER_ARM_LENGTH*LOWER_ARM_LENGTH - y1*y1) / (2.0f * z0);
    float b = (y1 - y0p) / z0;

    // discriminant
    // d = -(a + b*y1)^2 + rf*(b^2*rf + rf)
    float term = (a + b * y1);
    float d = -(term * term) + UPPER_ARM_LENGTH * (b*b*UPPER_ARM_LENGTH + UPPER_ARM_LENGTH);

    if (d < 0.0f) {
        return -1;
    }

    float sqrt_d = sqrtf(d);

    // elbow-down configuration: use -sqrt(d)
    float yj = (y1 - a*b - sqrt_d) / (b*b + 1.0f);
    float zj = a + b * yj;

    // theta = -pi - atan2(zj, (yj - y1))
    *theta = -(float)M_PI - atan2f(zj, (yj - y1));
    *theta = *theta * (180.0f / (float)M_PI);

    return 0;
}

// DELTA ROBOT INVERSE KINEMATICS
// Returns 0 if successful, -1 if unreachable
// Fills in t1, t2, t3 with motor angles in degrees
int IK(float x0, float y0, float z0, float *t1, float *t2, float *t3) {

    float theta1, theta2, theta3;
    int ok;

    // Arm 1 (no rotation)
    ok = calc_angleYZ(x0, y0, z0, &theta1);
    if (ok != 0) return -1;

    // Arm 2: rotate +120 deg
    float x1 = x0 * COS120 - y0 * SIN120;
    float y1 = y0 * COS120 + x0 * SIN120;
    ok = calc_angleYZ(x1, y1, z0, &theta2);
    if (ok != 0) return -1;

    // Arm 3: rotate -120 deg
    float x2 = x0 * COS120 + y0 * SIN120;  // cos(-120) = -0.5, sin(-120) = -0.866
    float y2 = y0 * COS120 - x0 * SIN120;
    ok = calc_angleYZ(x2, y2, z0, &theta3);
    if (ok != 0) return -1;

    *t1 = theta1;
    *t2 = theta2;
    *t3 = theta3;

    return 0;
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

// Stop all motion immediately
void stop_motion(void) {
    // TODO: Partner to implement motor disable/brake logic
    // For now, signal that motion should stop
}

// Set robot to idle state (stop motors)
void set_idle(robot_t *robot) {
    stop_motion();
    robot->flag_ready_to_move = false;
}

// Enter fault mode (safe state)
void safety_enter_fault_mode(void) {
    // TODO: Partner to implement fault handling:
    // - Disable all motors
    // - Engage brakes if available
    // - Set appropriate safety signals
}

