#pragma once

#include <stdint.h>
#include "ringBuffer.h"

// Functions sent by host
#define Set_Origin              0x00
#define Go_Absolute_Pos         0x01
#define Go_Relative_Pos         0x03
#define General_Read            0x0e
#define Read_Drive_Config       0x08
#define Set_MainGain            0x10
#define Set_SpeedGain           0x11
#define Set_IntGain             0x12
#define Set_HighSpeed           0x14
#define Set_HighAccel           0x15
#define Set_Pos_OnRange         0x16
#define Set_GearNumber          0x17
#define Read_MainGain           0x18
#define Read_HighSpeed          0x1c

// Functions sent by DYN drive
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
#define Is_AbsPos32             0x1b
#define Is_TrqCurrent           0x1e
#define Turn_ConstSpeed         0x0a

#define IO_MOTOR_COM_INPUT_BUFFER_LEN 256
#define IO_MOTOR_COM_OUTPUT_BUFFER_LEN 256
#define IO_MOTOR_COM_READ_PACKAGE_LEN 8

typedef struct {
  long motor_pos32;
  long motor_torque_current;
  long motor_speed32;

  uint8_t motor_position_ready_flag;
  uint8_t motor_torque_ready_flag;
  uint8_t motor_speed_ready_flag;

  uint8_t driver_main_gain;
  uint8_t driver_speed_gain;
  uint8_t driver_int_gain;
  uint8_t driver_trq_cons;
  uint8_t driver_high_speed;
  uint8_t driver_high_accel;
  uint8_t driver_read_id;
  uint8_t driver_status;
  uint8_t driver_on_range;
  long driver_config;
  uint8_t driver_config_ready_flag;
} io_motor_com_data_t;

typedef struct {
  ringBuffer_t input_buffer;
  ringBuffer_t output_buffer;
  unsigned char read_package_buffer[IO_MOTOR_COM_READ_PACKAGE_LEN];
  uint8_t read_num;
  uint8_t read_package_length;
  io_motor_com_data_t data;
} io_motor_com_t;

void io_motor_com_init(io_motor_com_t *ctx, uint8_t *input_storage, uint16_t input_size,
                       uint8_t *output_storage, uint16_t output_size);
void io_motor_com_read_package(io_motor_com_t *ctx);
void io_motor_com_get_function(io_motor_com_t *ctx);
int32_t io_motor_com_cal_sign_value(unsigned char *one_package);
long io_motor_com_cal_value(unsigned char *one_package);
void io_motor_com_send_package(io_motor_com_t *ctx, char id, long displacement, unsigned char function_code);
void io_motor_com_make_crc_send(io_motor_com_t *ctx, unsigned char length, unsigned char *buffer);

long io_motor_com_get_motor_pos(const io_motor_com_t *ctx);
long io_motor_com_get_motor_speed(const io_motor_com_t *ctx);
long io_motor_com_get_motor_torque(const io_motor_com_t *ctx);
long io_motor_com_get_driver_config(const io_motor_com_t *ctx);

uint8_t io_motor_com_get_motor_position_ready(const io_motor_com_t *ctx);
uint8_t io_motor_com_get_motor_speed_ready(const io_motor_com_t *ctx);
uint8_t io_motor_com_get_motor_torque_ready(const io_motor_com_t *ctx);
uint8_t io_motor_com_get_driver_config_ready(const io_motor_com_t *ctx);

void io_motor_com_set_motor_position_ready(io_motor_com_t *ctx, uint8_t value);
void io_motor_com_set_motor_speed_ready(io_motor_com_t *ctx, uint8_t value);
void io_motor_com_set_motor_torque_ready(io_motor_com_t *ctx, uint8_t value);
void io_motor_com_set_driver_config_ready(io_motor_com_t *ctx, uint8_t value);

// Lightweight RX diagnostics to debug motor comm timeouts.
void io_motor_com_debug_snapshot(uint32_t *rx_packets,
                                 uint32_t *rx_crc_bad,
                                 uint32_t *rx_pos_packets,
                                 uint8_t *last_id,
                                 uint8_t *last_func,
                                 uint8_t *last_crc_ok);
