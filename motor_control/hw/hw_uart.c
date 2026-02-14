#include "hw_uart.h"
#include "main.h"

HAL_StatusTypeDef hw_motor_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms)
{
    return HAL_UART_Transmit(&huart1, (uint8_t *)data, len, timeout_ms);
}

HAL_StatusTypeDef hw_motor_rx(uint8_t *data, uint16_t len, uint32_t timeout_ms)
{
    return HAL_UART_Receive(&huart1, data, len, timeout_ms);
}

HAL_StatusTypeDef hw_laptop_tx(const uint8_t *data, uint16_t len, uint32_t timeout_ms)
{
    return HAL_UART_Transmit(&huart2, (uint8_t *)data, len, timeout_ms);
}
