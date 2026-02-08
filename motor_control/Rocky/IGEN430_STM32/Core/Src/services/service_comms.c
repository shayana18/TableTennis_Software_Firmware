/*
 * service_comms.c
 *
 *  Created on: Feb 7, 2026
 *      Author: rocky
 */

#include "services/service_comms.h"
#include "robot.h"
#include <string.h>
#include <stdio.h>

// ============================================================
// CIRCULAR BUFFER FOR TARGET QUEUE
// ============================================================

#define COMMS_RX_BUFFER_SIZE 256
#define TARGET_QUEUE_SIZE 10

typedef struct {
    target_t targets[TARGET_QUEUE_SIZE];
    uint8_t write_idx;
    uint8_t read_idx;
    uint8_t count;
} target_queue_t;

// Global state
static uint8_t rx_buffer[COMMS_RX_BUFFER_SIZE];
static uint16_t rx_write_idx = 0;
static target_queue_t target_queue = {0};

// Note: UART handles live in main.c; transmission is left as a TODO for partner

// ============================================================
// TARGET QUEUE MANAGEMENT
// ============================================================

static void queue_push_target(target_t *target) {
    if (target_queue.count >= TARGET_QUEUE_SIZE) {
        return;  // Queue full, drop new target
    }
    
    target_queue.targets[target_queue.write_idx] = *target;
    target_queue.write_idx = (target_queue.write_idx + 1) % TARGET_QUEUE_SIZE;
    target_queue.count++;
}

static bool queue_pop_target(target_t *target) {
    if (target_queue.count == 0) {
        return false;
    }
    
    *target = target_queue.targets[target_queue.read_idx];
    target_queue.read_idx = (target_queue.read_idx + 1) % TARGET_QUEUE_SIZE;
    target_queue.count--;
    return true;
}

// ============================================================
// MESSAGE PARSING
// ============================================================

// Simple protocol: "x,y,z,t_arrival,type\n"
// Example: "100.5,50.2,-300.0,2.5,0\n"
// Type: 0=NONE, 1=INTERCEPT, 2=STRIKE, 3=HOME
static bool parse_target_message(const char *msg, target_t *target) {
    float x, y, z, t_arrival;
    int type_int;
    
    int parsed = sscanf(msg, "%f,%f,%f,%f,%d", &x, &y, &z, &t_arrival, &type_int);
    if (parsed != 5) {
        return false;
    }
    
    target->pos.x = x;
    target->pos.y = y;
    target->pos.z = z;
    target->t_arrival = t_arrival;
    target->type = (target_type)type_int;
    target->timestamp = 0;  // Could add timestamp from timer
    
    return true;
}

// Find newline in buffer and extract message
static bool extract_message(char *msg_buf, size_t msg_size) {
    // Find newline character
    for (uint16_t i = 0; i < rx_write_idx; i++) {
        if (rx_buffer[i] == '\n') {
            // Found complete message
            if (i + 1 > msg_size) {
                return false;  // Message too long
            }
            
            // Copy message (without newline)
            memcpy(msg_buf, rx_buffer, i);
            msg_buf[i] = '\0';
            
            // Remove message from buffer
            rx_write_idx -= (i + 1);
            memmove(rx_buffer, &rx_buffer[i + 1], rx_write_idx);
            
            return true;
        }
    }
    return false;  // No complete message yet
}

// ============================================================
// PUBLIC API
// ============================================================

/**
 * Process incoming UART data and update target queue
 * Call this from UART RX callback or main loop
 */
void comms_process_data(void) {
    char msg_buf[64];
    
    // Process all complete messages in buffer
    while (extract_message(msg_buf, sizeof(msg_buf))) {
        target_t new_target;
        
        if (parse_target_message(msg_buf, &new_target)) {
            queue_push_target(&new_target);
        }
    }
}

/**
 * Get next available target from queue
 * Returns true if target was available, false if queue empty
 */
bool comms_pop_new_target(target_t *t) {
    return queue_pop_target(t);
}

/**
 * Send status message to PC
 * Non-blocking if using UART DMA
 */
void comms_send_status(const char *msg) {
    // TODO: Partner to implement actual UART transmission
    // For now, this is a stub
    
    // Example implementation (blocking):
    // HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
    
    // Or DMA/interrupt-based:
    // HAL_UART_Transmit_IT(&huart2, (uint8_t*)msg, strlen(msg));
}

/**
 * UART RX callback - called when data arrives
 * Partner to integrate this with HAL_UART_RxCpltCallback
 */
void comms_uart_rx_callback(uint8_t data) {
    if (rx_write_idx < COMMS_RX_BUFFER_SIZE) {
        rx_buffer[rx_write_idx++] = data;
    }
}

/**
 * Get current rx buffer fill level (for debugging)
 */
uint16_t comms_get_rx_buffer_level(void) {
    return rx_write_idx;
}

/**
 * Clear rx buffer (for debugging/recovery)
 */
void comms_clear_rx_buffer(void) {
    rx_write_idx = 0;
    memset(rx_buffer, 0, COMMS_RX_BUFFER_SIZE);
}


