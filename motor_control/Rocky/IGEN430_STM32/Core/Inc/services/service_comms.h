/*
 * service_comms.h
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#ifndef INC_SERVICES_SERVICE_COMMS_H_
#define INC_SERVICES_SERVICE_COMMS_H_

#include <stdint.h>
#include <stdbool.h>
#include "robot.h"

// Get next target from queue (returns true if available)
bool comms_pop_new_target(target_t *t);

// Send status message to PC (non-blocking if possible)
void comms_send_status(const char *msg);

// Process incoming UART data (call from main loop or ISR)
void comms_process_data(void);

// UART RX callback - call this from HAL_UART_RxCpltCallback
void comms_uart_rx_callback(uint8_t data);

// Debug functions
uint16_t comms_get_rx_buffer_level(void);
void comms_clear_rx_buffer(void);

#endif /* INC_SERVICES_SERVICE_COMMS_H_ */
