#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "message.h"

#define MAX_BUF_SIZE 30


typedef struct ringBuffer
{
    target_t buf[MAX_BUF_SIZE];
    volatile uint8_t head;
    volatile uint8_t tail;
}ringBuffer_t; 

typedef enum bufErrCodes_e
{
    ENQUEUE_OK,
    DEQUEUE_OK, 
    EMPTY, 
    FULL_OVERWRITE,
} buffErrCodes_e; 

void initRingBuffer(ringBuffer_t *buffer);

// input must point to TARGET_MSG_FLOAT_COUNT floats
buffErrCodes_e enqueue(ringBuffer_t *buffer, const float *input);

// output must point to TARGET_MSG_FLOAT_COUNT floats
buffErrCodes_e dequeue(ringBuffer_t *buffer, float *output); 

bool bufIsEmpty(ringBuffer_t *buffer); 


