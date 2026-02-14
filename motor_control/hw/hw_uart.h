#pragma once

#include <stdint.h>
#include "stm32c0xx_hal.h"

HAL_StatusTypeDef hw_uart1_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms);
HAL_StatusTypeDef hw_uart1_rx(uint8_t *data, uint16_t len, uint32_t timeout_ms);
HAL_StatusTypeDef hw_uart2_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms);
