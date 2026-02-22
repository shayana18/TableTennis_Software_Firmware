#include "io_motor_com.h"
#include "main.h"
#include "hw_uart.h"

#include <stdint.h>

static volatile uint32_t s_dbg_rx_packets = 0;
static volatile uint32_t s_dbg_rx_crc_bad = 0;
static volatile uint32_t s_dbg_rx_pos_packets = 0;
static volatile uint8_t s_dbg_last_id = 0;
static volatile uint8_t s_dbg_last_func = 0;
static volatile uint8_t s_dbg_last_crc_ok = 0;


void io_motor_com_init(io_motor_com_t *ctx, uint8_t *input_storage, uint16_t input_size,
                       uint8_t *output_storage, uint16_t output_size)
{
  initRingBuffer(&ctx->input_buffer, input_storage, input_size);
  initRingBuffer(&ctx->output_buffer, output_storage, output_size);
  ctx->read_num = 0;
  ctx->read_package_length = 0;
  ctx->data.driver_config = 0;
  ctx->data.driver_config_ready_flag = 0x00;
}

/* Function Name: classifyReadString
 * Author: Aidan Drescher
 * Date: 2024-04-11
 * Purpose: Reads data from drive.
 * */
void io_motor_com_read_package(io_motor_com_t *ctx) {

  uint8_t c;
  int cif;

  while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE)) {
    (void)hw_motor_rx(&c, 1, 0); // Receive one byte from UART
    enqueue(&ctx->input_buffer, c);
  }

  while (!bufIsEmpty(&ctx->input_buffer)) {
    uint8_t byte = 0;
    if (dequeue(&ctx->input_buffer, &byte) != DEQUEUE_OK) {
      break;
    }
    c = byte;
    cif = c & 0x80;
    if (cif == 0) {
      ctx->read_num = 0;
      ctx->read_package_length = 0;
    }
    if (cif == 0 || ctx->read_num > 0) {
      ctx->read_package_buffer[ctx->read_num] = c;
      ctx->read_num++;
      if (ctx->read_num == 2) {
        cif = c >> 5;
        cif = cif & 0x03;
        ctx->read_package_length = 4 + cif;
        c = 0;
      }
      if (ctx->read_num == ctx->read_package_length) {
        io_motor_com_get_function(ctx);
        ctx->read_num = 0;
        ctx->read_package_length = 0;
      }
    }
  }
}

/* Function Name: Get_Function
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Decodes the desired function code and toggles related flag. Initiates data processing
 * */
void io_motor_com_get_function(io_motor_com_t *ctx) {
  char ID, ReceivedFunction_Code, CRC_Check;
  long Temp32;
  ID = ctx->read_package_buffer[0] & 0x7f;
  ReceivedFunction_Code = ctx->read_package_buffer[1] & 0x1f;
  s_dbg_rx_packets++;
  s_dbg_last_id = (uint8_t)ID;
  s_dbg_last_func = (uint8_t)ReceivedFunction_Code;
  CRC_Check = 0;

  for (int i = 0; i < ctx->read_package_length - 1; i++) {
    CRC_Check += ctx->read_package_buffer[i];
  }

  CRC_Check ^= ctx->read_package_buffer[ctx->read_package_length - 1];
  CRC_Check &= 0x7f;
  if (CRC_Check != 0) {
    s_dbg_rx_crc_bad++;
    s_dbg_last_crc_ok = 0;
    return;
  }
  else {
    s_dbg_last_crc_ok = 1;
    if (ReceivedFunction_Code == Is_AbsPos32) {
      s_dbg_rx_pos_packets++;
    }
    switch (ReceivedFunction_Code)
    {
      case  Is_AbsPos32:
        ctx->data.motor_pos32 = io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.motor_position_ready_flag = 0x00;
        break;
    //   case  Is_MotorSpeed:
    //     Motor_Speed32 = Cal_SignValue(Read_Package_Buffer);
    //     MotorSpeed32Ready_Flag = 0x00;
    //     break;
      case  Is_TrqCurrent:
        ctx->data.motor_torque_current = io_motor_com_cal_sign_value(ctx->read_package_buffer) ;
        ctx->data.motor_torque_current_ready_flag = 0x00;
        break;
      case  Is_Status:
        ctx->data.driver_status = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        // Driver_Status=drive status byte data
        break;
      case  Is_Config:
        Temp32 = io_motor_com_cal_value(ctx->read_package_buffer);
        ctx->data.driver_config = Temp32;
        ctx->data.driver_config_ready_flag = 0x00;
        break;
      case  Is_MainGain:
        ctx->data.driver_main_gain = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_main_gain = ctx->data.driver_main_gain & 0x7f;
        break;
      case  Is_SpeedGain:
        ctx->data.driver_speed_gain = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_speed_gain = ctx->data.driver_speed_gain & 0x7f;
        break;
      case  Is_IntGain:
        ctx->data.driver_int_gain = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_int_gain = ctx->data.driver_int_gain & 0x7f;
        break;
      case  Is_TrqCons:
      {
        const long trq_cons = io_motor_com_cal_value(ctx->read_package_buffer);
        ctx->data.motor_torque_const = trq_cons;
        ctx->data.driver_trq_cons = (uint8_t)(trq_cons & 0x7f);
        ctx->data.motor_torque_const_ready = 0x00;
        break;
      }
      case  Is_HighSpeed:
        ctx->data.driver_high_speed = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_high_speed = ctx->data.driver_high_speed & 0x7f;
        break;
      case  Is_HighAccel:
        ctx->data.driver_high_accel = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_high_accel = ctx->data.driver_high_accel & 0x7f;
        break;
      case  Is_Driver_ID:
        ctx->data.driver_read_id = ID;
        break;
      case  Is_Pos_OnRange:
        ctx->data.driver_on_range = (char)io_motor_com_cal_sign_value(ctx->read_package_buffer);
        ctx->data.driver_on_range = ctx->data.driver_on_range & 0x7f;
        break;
    }
  }
}
/* Function Name: Cal_SignValue
 * Author: Aidan Drescher
 * Date: 2024-04-17
 * Purpose: Interprets the correct mathematical sign (pos/neg) from the 2^18 value
 * (splits into -2^17 ~ 2^17 - 1)
 * */
int32_t io_motor_com_cal_sign_value(unsigned char* One_Package) {
    char Package_Length, i;
    int32_t Lcmd;

    // Determine package length based on the second byte
    Package_Length = 4 + ((One_Package[1] >> 5) & 0x03);

    // Initialize the command value
    Lcmd = (One_Package[2] & 0x7F); // Mask out the sign bit

    // Extract the sign bit from the first byte
    int sign_bit = (One_Package[2] & 0x40) ? 1 : 0;

    // Sign extension if necessary
    if (sign_bit) {
        Lcmd |= 0xFFFFFF80; // Sign extend to 32 bits
    }

    // Process the remaining bytes
    for (i = 3; i < Package_Length - 1; i++) {
        Lcmd = (Lcmd << 7) | (One_Package[i] & 0x7F);
    }

    return Lcmd; // Lcmd: -2^17 ~ 2^17 - 1
}

/* Function Name: Cal_Value
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose:
 * */
long io_motor_com_cal_value(unsigned char* One_Package)
{
  char Package_Length,OneChar,i;
  long Lcmd;
  OneChar = One_Package[1];
  OneChar = OneChar>>5;
  OneChar = OneChar&0x03;
  Package_Length = 4 + OneChar;

  OneChar = One_Package[2];   /*First byte 0x7f, bit 6 reprents sign      */
  OneChar &= 0x7f;
  Lcmd = (long)OneChar;     /*Sign extended to 32bits           */
  for(i=3;i<Package_Length-1;i++)
  {
    OneChar = One_Package[i];
    OneChar &= 0x7f;
    Lcmd = Lcmd<<7;
    Lcmd += OneChar;
  }
  return(Lcmd);         /*Lcmd : -2^27 ~ 2^27 - 1           */
}

/* Function Name: Send_Package
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares and sends a data packet
 * */
void io_motor_com_send_package(io_motor_com_t *ctx, char ID , long Displacement, unsigned char function_code) {
  unsigned char B[8], Package_Length, Function_Code;
  long TempLong;
  B[1] = B[2] = B[3] = B[4] = B[5] = (unsigned char)0x80;
  B[0] = ID & 0x7f;
  Function_Code = function_code & 0x1f;
  TempLong = Displacement & 0x0fffffff; //Max 28bits
  B[5] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[4] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[3] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[2] += (unsigned char)TempLong & 0x0000007f;
  Package_Length = 7;
  TempLong = Displacement;
  TempLong = TempLong >> 20;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //Three byte data
    B[2] = B[3];
    B[3] = B[4];
    B[4] = B[5];
    Package_Length = 6;
  }
  TempLong = Displacement;
  TempLong = TempLong >> 13;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //Two byte data
    B[2] = B[3];
    B[3] = B[4];
    Package_Length = 5;
  }
  TempLong = Displacement;
  TempLong = TempLong >> 6;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //One byte data
    B[2] = B[3];
    Package_Length = 4;
  }
  B[1] += (Package_Length - 4) * 32 + Function_Code;

  io_motor_com_make_crc_send(ctx, Package_Length, B);
}

/* Function Name: Make_CRC_Send
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Checks packet and transmits
 * */
void io_motor_com_make_crc_send(io_motor_com_t *ctx, unsigned char Plength, unsigned char* B) {
  unsigned char Error_Check = 0;
  char RS232_HardwareShiftRegister;

  for (int i = 0; i < Plength - 1; i++) {
    enqueue(&ctx->output_buffer, B[i]);
    Error_Check += B[i];
  }
  Error_Check = Error_Check | 0x80;
  enqueue(&ctx->output_buffer, Error_Check);
  while (!bufIsEmpty(&ctx->output_buffer)) {
    uint8_t byte = 0;
    if (dequeue(&ctx->output_buffer, &byte) != DEQUEUE_OK) {
      break;
    }
    RS232_HardwareShiftRegister = (char)byte;
    hw_motor_tx((uint8_t *)&RS232_HardwareShiftRegister, 1, HAL_MAX_DELAY);
  }
}

/***************************** GETTER AND SETTERS FOR FLAGS ***************************/

long io_motor_com_get_motor_pos(const io_motor_com_t *ctx)
{
  return ctx->data.motor_pos32;
}

long io_motor_com_get_motor_speed(const io_motor_com_t *ctx)
{
  return ctx->data.motor_speed32;
}

long io_motor_com_get_motor_torque_current(const io_motor_com_t *ctx)
{
  return ctx->data.motor_torque_current;
}

long io_motor_com_get_motor_torque_const(const io_motor_com_t *ctx)
{
  return ctx->data.motor_torque_const;
}

long io_motor_com_get_driver_config(const io_motor_com_t *ctx)
{
  return ctx->data.driver_config;
}

uint8_t io_motor_com_get_motor_position_ready(const io_motor_com_t *ctx)
{
  return ctx->data.motor_position_ready_flag;
}

uint8_t io_motor_com_get_motor_speed_ready(const io_motor_com_t *ctx)
{
  return ctx->data.motor_speed_ready_flag;
}

uint8_t io_motor_com_get_motor_torque_ready(const io_motor_com_t *ctx)
{
  return ctx->data.motor_torque_ready_flag;
}

uint8_t io_motor_com_get_torque_current_ready(const io_motor_com_t *ctx)
{
  return ctx->data.motor_torque_current_ready_flag; 
}

uint8_t io_motor_com_get_torque_const_ready(const io_motor_com_t *ctx)
{
  return ctx->data.motor_torque_const_ready; 
}

uint8_t io_motor_com_get_driver_config_ready(const io_motor_com_t *ctx)
{
  return ctx->data.driver_config_ready_flag;
}

void io_motor_com_set_motor_position_ready(io_motor_com_t *ctx, uint8_t value)
{
  ctx->data.motor_position_ready_flag = value;
}

void io_motor_com_set_motor_speed_ready(io_motor_com_t *ctx, uint8_t value)
{
  ctx->data.motor_speed_ready_flag = value;
}

void io_motor_com_set_motor_torque_ready(io_motor_com_t *ctx, uint8_t value)
{
  ctx->data.motor_torque_ready_flag = value;
}

void io_motor_com_set_torque_current_ready(io_motor_com_t *ctx, uint8_t value)
{
    ctx->data.motor_torque_current_ready_flag = value;
}

void io_motor_com_set_torque_const_ready(io_motor_com_t *ctx, uint8_t value)
{
    ctx->data.motor_torque_const_ready = value;
}

void io_motor_com_set_driver_config_ready(io_motor_com_t *ctx, uint8_t value)
{
  ctx->data.driver_config_ready_flag = value;
}


void io_motor_com_debug_snapshot(uint32_t *rx_packets,
                                 uint32_t *rx_crc_bad,
                                 uint32_t *rx_pos_packets,
                                 uint8_t *last_id,
                                 uint8_t *last_func,
                                 uint8_t *last_crc_ok)
{
  if (rx_packets) {
    *rx_packets = s_dbg_rx_packets;
  }
  if (rx_crc_bad) {
    *rx_crc_bad = s_dbg_rx_crc_bad;
  }
  if (rx_pos_packets) {
    *rx_pos_packets = s_dbg_rx_pos_packets;
  }
  if (last_id) {
    *last_id = s_dbg_last_id;
  }
  if (last_func) {
    *last_func = s_dbg_last_func;
  }
  if (last_crc_ok) {
    *last_crc_ok = s_dbg_last_crc_ok;
  }
}
