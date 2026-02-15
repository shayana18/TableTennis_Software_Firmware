#pragma once

#include <stdint.h>
#include "stm32c0xx_hal.h"

HAL_StatusTypeDef hw_motor_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms);
HAL_StatusTypeDef hw_motor_rx(uint8_t *data, uint16_t len, uint32_t timeout_ms);
HAL_StatusTypeDef hw_laptop_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms);
