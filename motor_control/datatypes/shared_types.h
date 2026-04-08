#pragma once

#include <stdint.h>

// Float-array message layout: type is stored as a float when using float-only buffers.
#define TARGET_MSG_FLOAT_COUNT 10
// TO FILL IN WITH REAL VALUES

typedef enum {
  TARGET_NONE = 0,
  TARGET_INTERCEPT = 1,
  TARGET_STRIKE = 2,
  TARGET_HOME = 3,
  TARGET_TEST = 4
} target_type;

typedef enum {
  TARGET_MSG_TYPE = 0,
  TARGET_MSG_X_POS,
  TARGET_MSG_Y_POS,
  TARGET_MSG_Z_POS,
  TARGET_MSG_X_VEL,
  TARGET_MSG_Y_VEL, 
  TARGET_MSG_Z_VEL,
  TARGET_MSG_INTERCEPT_TIME,
  TARGET_MSG_TIME_SENT,
  TARGET_MSG_TIMESTAMP
} target_msg_field_t;

typedef struct {
  float x;   					// (mm)
  float y;						// (mm)
  float z;						// (mm)
} vec3;

typedef struct {
  target_type type;
  vec3 intercept_pos; 
  vec3 intercept_vel;        // (mm/s)
  float intercept_time;         // (time in which ball will be at pos)
  float time_sent; 				// (time message was put on uart from laptop)
  float timestamp;				// (time frame arrived to camera)
  uint32_t received_time;		// local robot tick at UART reception (ms), not from message payload
} target_t;
