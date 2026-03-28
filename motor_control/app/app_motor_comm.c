#include "app_motor_comm.h"

#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "hw_uart.h"
#include "io_motor_com.h"
#include "main.h"

#define MOTOR_POS_READ_TIMEOUT_MS 200U
#define MOTOR_POS_READ_RETRIES 5U
#define MOTOR_POS_RETRY_GAP_MS 2U
#define MOTOR_UART_RECOVER_DEBUG 0U

#define TORQUE_CURRENT_READING_MAX 32767
#define MOTOR_RMS_CURRENT 8.4f

static void motor_uart_recover_and_drain(void)
{
  uint8_t dump = 0;
  const uint8_t had_ore = (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_ORE) != RESET);
  const uint8_t had_fe = (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_FE) != RESET);
  const uint8_t had_ne = (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_NE) != RESET);

#if MOTOR_UART_RECOVER_DEBUG
  if (had_ore) {
    static const char msg[] = "UART1 RX ERR: ORE\r\n";
    hw_laptop_tx((const uint8_t *)msg, (uint16_t)(sizeof(msg) - 1U), 20U);
  }
  if (had_fe) {
    static const char msg[] = "UART1 RX ERR: FE\r\n";
    hw_laptop_tx((const uint8_t *)msg, (uint16_t)(sizeof(msg) - 1U), 20U);
  }
  if (had_ne) {
    static const char msg[] = "UART1 RX ERR: NE\r\n";
    hw_laptop_tx((const uint8_t *)msg, (uint16_t)(sizeof(msg) - 1U), 20U);
  }
#else
  (void)had_ore;
  (void)had_fe;
  (void)had_ne;
#endif

  __HAL_UART_CLEAR_OREFLAG(&huart1);
  __HAL_UART_CLEAR_FEFLAG(&huart1);
  __HAL_UART_CLEAR_NEFLAG(&huart1);

  while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE)) {
    (void)hw_motor_rx(&dump, 1, 0);
  }
}

/* Function Name: clockwiseMotion
 * Purpose: Send a command packet to DMM drive to initiate clockwise motion of the shaft with user
 * defined speed
 * */
void clockwiseMotion(io_motor_com_t *motor_com, char motor_id, int speed, char motor_direction){
	(void)motor_direction;
	// TODO: gate motion on motor power state to avoid commanding a disabled drive.
	int send_speed = speed;
	Turn_const_speed(motor_com, motor_id, send_speed);
	// TODO: caller should update motor_direction to 1 (CW) if tracking direction state.
}

/* Function Name: counterClockwiseMotion
 * Purpose: Send a command packet to DMM drive to initiate counter-clockwise motion of the shaft
 * with user defined speed
 * */
void counterClockwiseMotion(io_motor_com_t *motor_com, char motor_id, int speed, char motor_direction){
	(void)motor_direction;
	// TODO: gate motion on motor power state to avoid commanding a disabled drive.
	int send_speed = speed*-1;
	Turn_const_speed(motor_com, motor_id, send_speed);
	// TODO: caller should update motor_direction to 2 (CCW) if tracking direction state.
}

/* Function Name: speedControl
 * Purpose: Parse incoming speed commands to separate the command type and the real value. Change
 * the multiplier to adjust the scale of the speed
 * */
void speedControl(io_motor_com_t *motor_com, char motor_id, char* command, int *speed, char motor_direction){
  int index = 0;
  int i = 0;
  char speedVal[3] = {};
  for(i=0; command[i]!='\0'; i++){
	  if(isdigit(command[i])){
	      speedVal[index] = command[i];
	      index++;
	  }
  }
  *speed = atoi(speedVal);
  *speed = *speed * 10; //change multiplier here to determine max speed (motor rated 3000RPM - set to 30)

  // TODO: skip issuing speed commands when motor power is disabled.
  if(motor_direction == 1){
	  clockwiseMotion(motor_com, motor_id, *speed, motor_direction);
  }
  else if(motor_direction == 2){
	  counterClockwiseMotion(motor_com, motor_id, *speed, motor_direction);
  }
}

/* Function Name: absoluteMove
 * Purpose: Parse incoming absolute move commands to separate the command type and the real value.
 * The function computes the move time based on # of revolutions and calls appropriate functions
 * to execute the move (move_rel32)
 * */
void absoluteMove(io_motor_com_t *motor_com, char motor_id, char* command, char motor_direction){
	int index = 0;
	int i = 0;
	char revCount[3] = {};
	long int noRev = 0;
	int moveTime = 0;
	(void)motor_direction;
	// TODO: gate motion on motor power state to avoid commanding a disabled drive.
	// TODO: caller should update motor_direction to 0 (stopped) if tracking direction state.
	Turn_const_speed(motor_com, motor_id, 0);
	HAL_Delay(2000);
	//while timer is not done, wait (still run read data for position display)
	for(i=0; command[i]!='\0'; i++){
		if(isdigit(command[i])){
		    revCount[index] = command[i];
		    index++;
		}
	}
	noRev = atoi(revCount);
	moveTime = 550 * noRev;
	if(command[1] == '-'){
	    noRev = noRev * -65536;
		move_rel32(motor_com, motor_id, noRev);
		HAL_Delay(moveTime);
	}
	else{
		noRev = noRev * 65536;
		move_rel32(motor_com, motor_id, noRev);
		HAL_Delay(moveTime);
	}
}

/* Function Name: absMoveSetup
 * Author: Aidan Drescher
 * Date: 2024-04-10
 * Purpose: Ensures positional accuracy before relative moves
 * */
void absMoveSetup(char motor_id){
	(void)motor_id;

	for(int i=0; i<45; i++){
		//sendString();
		HAL_Delay(75);
	}
}

/* Function Name: goHome
 * Author: Aidan Drescher
 * Date: 2024-03-19
 * Purpose: Sends the motor back to 0 pulses on the encoder
 * */
int goHome(io_motor_com_t *motor_com, char motor_id, char motor_direction){
  (void)motor_direction;
  // TODO: gate homing on motor power state (return 5 if power is off).
  // TODO: caller should update motor_direction to 0 (stopped) if tracking direction state.

  Turn_const_speed(motor_com, motor_id, 0);
  ReadMotorPosition32(motor_com, motor_id);

  if(io_motor_com_get_motor_pos(motor_com) > 400000){
    while(io_motor_com_get_motor_pos(motor_com) > 400000){
      //sendString();
  	ReadMotorPosition32(motor_com, motor_id);
      Turn_const_speed(motor_com, motor_id, -750);
    }
    while(io_motor_com_get_motor_pos(motor_com) > 200000 && io_motor_com_get_motor_pos(motor_com) <= 400000){
      //sendString();
  	ReadMotorPosition32(motor_com, motor_id);
      Turn_const_speed(motor_com, motor_id, -250);
    }
    Turn_const_speed(motor_com, motor_id, 0);
    absMoveSetup(motor_id);
    ReadMotorPosition32(motor_com, motor_id);
    move_rel32(motor_com, motor_id, -io_motor_com_get_motor_pos(motor_com));
    return 1;
  }
  else if(io_motor_com_get_motor_pos(motor_com) < -400000){
    while(io_motor_com_get_motor_pos(motor_com) < -400000){
      //sendString();
  	ReadMotorPosition32(motor_com, motor_id);
      Turn_const_speed(motor_com, motor_id, 750);
    }
    while(io_motor_com_get_motor_pos(motor_com) < -200000 && io_motor_com_get_motor_pos(motor_com) >= -400000){
      //sendString();
  	ReadMotorPosition32(motor_com, motor_id);
      Turn_const_speed(motor_com, motor_id, 250);
    }
    Turn_const_speed(motor_com, motor_id, 0);
    absMoveSetup(motor_id);
    ReadMotorPosition32(motor_com, motor_id);
    move_rel32(motor_com, motor_id, -io_motor_com_get_motor_pos(motor_com));
    return 1;
  }
  else{
    Turn_const_speed(motor_com, motor_id, 0);
    absMoveSetup(motor_id);
    ReadMotorPosition32(motor_com, motor_id);
    move_rel32(motor_com, motor_id, -io_motor_com_get_motor_pos(motor_com));
    return 1;
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DYN DRIVE FUNCTIONS (DONT CHANGE)////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

/* Function Name: move_rel32
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares a relative move packet to send to DMM drive
 * */
void move_rel32(io_motor_com_t *motor_com, char ID, long pos)
{
  char Axis_Num = ID;
  io_motor_com_send_package(motor_com, Axis_Num, pos, (char)Go_Relative_Pos);
}

/* Function Name: ReadMotorTorqueCurrent
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Executes a read from the DMM drive to obtain motor torque
 * */

void ReadMotorTorqueConst(io_motor_com_t *motor_com, char ID)
{
  if (motor_com == NULL) {
    return;
  }

  for (uint8_t attempt = 0; attempt < MOTOR_POS_READ_RETRIES; attempt++) {
    motor_uart_recover_and_drain();

    // No dedicated read function code for torque constant; use General_Read + IS code.
    io_motor_com_send_package(motor_com, ID, Is_TrqCons, (char)General_Read);
    io_motor_com_set_torque_const_ready(motor_com, 0xff);
    const uint32_t t0 = HAL_GetTick();

    while ((HAL_GetTick() - t0) < MOTOR_POS_READ_TIMEOUT_MS) {
      io_motor_com_read_package(motor_com);
      if (io_motor_com_get_torque_const_ready(motor_com) == 0x00) {
        const long torque_const = io_motor_com_get_motor_torque_const(motor_com);
        char msg[64];
        snprintf(msg, sizeof(msg), "M%u torque_const=%ld\r\n", (unsigned char)ID, torque_const);
        hw_laptop_tx((const uint8_t *)msg, (uint16_t)strlen(msg), 20U);
        return;
      }
    }

    if ((attempt + 1U) < MOTOR_POS_READ_RETRIES) {
      HAL_Delay(MOTOR_POS_RETRY_GAP_MS);
    }
  }

  {
    char msg[48];
    snprintf(msg, sizeof(msg), "ERR: torque const timeout id=%u\r\n", (unsigned char)ID);
    hw_laptop_tx((const uint8_t *)msg, (uint16_t)strlen(msg), 20U);
  }
}

void ReadMotorTorqueCurrent(io_motor_com_t *motor_com, char ID)
{
  if (motor_com == NULL) {
    return;
  }

  for (uint8_t attempt = 0; attempt < MOTOR_POS_READ_RETRIES; attempt++) {
    motor_uart_recover_and_drain();

    // No dedicated read function code for torque current; use General_Read + IS code.
    io_motor_com_set_torque_current_ready(motor_com, 0xff);
    io_motor_com_send_package(motor_com, ID, Is_TrqCurrent, (char)General_Read);
    const uint32_t t0 = HAL_GetTick();

    while ((HAL_GetTick() - t0) < MOTOR_POS_READ_TIMEOUT_MS) {
      io_motor_com_read_package(motor_com);
      if (io_motor_com_get_torque_current_ready(motor_com) == 0x00) {
        const long raw = io_motor_com_get_motor_torque_current(motor_com);
        const float trq_ratio = (float)raw / (float)TORQUE_CURRENT_READING_MAX;
        const long current_ma = (long)(trq_ratio * (MOTOR_RMS_CURRENT * 1000.0f));
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "M%u raw=%ld current=%ldmA\r\n",
                 (unsigned char)ID, raw, current_ma);
        hw_laptop_tx((const uint8_t *)msg, (uint16_t)strlen(msg), 20U);
        return;
      }
    }

    if ((attempt + 1U) < MOTOR_POS_READ_RETRIES) {
      HAL_Delay(MOTOR_POS_RETRY_GAP_MS);
    }
  }

  {
    char msg[48];
    snprintf(msg, sizeof(msg), "ERR: torque timeout id=%u\r\n", (unsigned char)ID);
    hw_laptop_tx((const uint8_t *)msg, (uint16_t)strlen(msg), 20U);
  }
}

/* Function Name: ReadMotorPosition32
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Executes a read from the DMM drive to obtain motor position in encoder pulses
 * */
static bool read_motor_position32_impl(io_motor_com_t *motor_com, char ID, bool log_timeout)
{
  
  for (uint8_t attempt = 0; attempt < MOTOR_POS_READ_RETRIES; attempt++) {
    motor_uart_recover_and_drain();

    // No dedicated read function code for position; use General_Read + IS code.
    io_motor_com_send_package(motor_com, ID, Is_AbsPos32, (char)General_Read);
    io_motor_com_set_motor_position_ready(motor_com, 0xff);
    const uint32_t t0 = HAL_GetTick();

    while ((HAL_GetTick() - t0) < MOTOR_POS_READ_TIMEOUT_MS) {
      io_motor_com_read_package(motor_com);

      if (io_motor_com_get_motor_position_ready(motor_com) == 0x00) {
        return true;
      }
    }

    if ((attempt + 1U) < MOTOR_POS_READ_RETRIES) {
      HAL_Delay(MOTOR_POS_RETRY_GAP_MS);
    }
  }

  if (log_timeout) {
    char msg[64];
    snprintf(msg, sizeof(msg), "ERR: pos timeout id=%u\r\n", (unsigned char)ID);
    HAL_UART_Transmit(&huart2, (uint8_t *)msg, (uint16_t)strlen(msg), HAL_MAX_DELAY);
  }

  return false;
}

bool ReadMotorPosition32(io_motor_com_t *motor_com, char ID)
{
  return read_motor_position32_impl(motor_com, ID, true);
}

bool ReadMotorPosition32Quiet(io_motor_com_t *motor_com, char ID)
{
  return read_motor_position32_impl(motor_com, ID, false);
}

/* Function Name: ReadMotorSpeed32
 * Author: Aidan Drescher
 * Date: 2024-04-10
 * Purpose: Executes a read from the DMM drive to obtain motor speed in RPM
 * */

 // TODO: THIS IS WRONG 
void ReadMotorSpeed32(io_motor_com_t *motor_com, char ID)
{
  // Dedicated read function code for high speed setting (1 dummy byte).
  io_motor_com_send_package(motor_com, ID, 0, (char)Read_HighSpeed);
  io_motor_com_set_motor_speed_ready(motor_com, 0xff);
//   while(io_motor_com_get_motor_speed_ready(motor_com) != 0x00)
//   {
    // io_motor_com_read_package(motor_com);
//   }
}

/* Function Name: ReadDriverConfig
 * Purpose: Executes a read from the DMM drive to obtain drive configuration
 * */
void ReadDriverConfig(io_motor_com_t *motor_com, char ID)
{
  io_motor_com_set_driver_config_ready(motor_com, 0xff);
  io_motor_com_send_package(motor_com, ID, 0x01, (char)0x06);

  while (io_motor_com_get_driver_config_ready(motor_com) != 0x00)
  {
    io_motor_com_read_package(motor_com);
  }
}
/* Function Name: ReadMotorTorqueCurrent
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares an absolute move packet to send to DMM drive
 * */
void move_abs32(io_motor_com_t *motor_com, char MotorID, long Pos32)
{
  char Axis_Num = MotorID;
  io_motor_com_send_package(motor_com, Axis_Num, Pos32, (char)Go_Absolute_Pos);
}

/* Function Name: Turn_const_speed
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares and sends a data packet to execute a constant speed command
 * */
void Turn_const_speed(io_motor_com_t *motor_com, char ID, long spd)
{
      char Axis_Num = ID;
      io_motor_com_send_package(motor_com, Axis_Num, spd, Turn_ConstSpeed);
}
