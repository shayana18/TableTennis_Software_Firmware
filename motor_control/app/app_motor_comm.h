#pragma once

#include <stdbool.h>

#include "io_motor_com.h"

void clockwiseMotion(io_motor_com_t *motor_com, char motor_id, int speed, char motor_direction);
void counterClockwiseMotion(io_motor_com_t *motor_com, char motor_id, int speed, char motor_direction);
void speedControl(io_motor_com_t *motor_com, char motor_id, char *command, int *speed, char motor_direction);
void absoluteMove(io_motor_com_t *motor_com, char motor_id, char *command, char motor_direction);
void absMoveSetup(char motor_id);
int goHome(io_motor_com_t *motor_com, char motor_id, char motor_direction);

void move_rel32(io_motor_com_t *motor_com, char id, long pos);
void ReadMotorTorqueConst(io_motor_com_t *motor_com, char id);
void ReadMotorTorqueCurrent(io_motor_com_t *motor_com, char id);
bool ReadMotorPosition32(io_motor_com_t *motor_com, char id);
bool ReadMotorPosition32Quiet(io_motor_com_t *motor_com, char id);
void ReadMotorSpeed32(io_motor_com_t *motor_com, char id);
void ReadDriverConfig(io_motor_com_t *motor_com, char id);
void move_abs32(io_motor_com_t *motor_com, char motor_id, long pos32);
void Turn_const_speed(io_motor_com_t *motor_com, char id, long spd);
