/*
 * DMM_Driver.C
 *
 * 	Core driver functions for the DMM DYN2 drivers.
 *
 *  Created on: Jan 25, 2026
 *      Author: rocky
 */

/* Includes ------------------------------------------------------------------*/
//#include "DMM_driver.h"
#include <ctype.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h> // For sleep() function


#define Go_Absolute_Pos         0x01
#define Go_Relative_Pos         0x03
#define Is_AbsPos32             0x1b
#define General_Read            0x0e
#define Is_TrqCurrent           0x1E
#define Read_MainGain           0x18
#define Set_MainGain            0x10
#define Set_SpeedGain           0x11
#define Set_IntGain             0x12
#define Set_HighSpeed           0x14
#define Set_HighAccel           0x15
#define Set_Pos_OnRange         0x16
#define Is_MainGain             0x10
#define Is_SpeedGain            0x11
#define Is_IntGain              0x12
#define Is_TrqCons              0x13
#define Is_HighSpeed            0x14
#define Is_HighAccel            0x15
#define Is_Driver_ID            0x16
#define Is_Pos_OnRange          0x17
#define Is_Status               0x19
#define Is_Config               0x1a
#define Is_MotorSpeed			0x1d
#define Is_DriveReset			0x1c
#define Is_DriveEnable			0x20
#define Is_DriveDisable			0x21
#define UART2_BAUD				115200 //Set Baud Rate for USB Communication (To Tablet or Computer)
#define UART3_BAUD				38400 //Set Baud Rate for DYN Drive Communication (Nominally 38400)



