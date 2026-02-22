#pragma once

#include <stdint.h>
#include <stdbool.h>

typedef struct ringBuffer
{
    uint8_t *buf;
    uint16_t size;
    volatile uint16_t head;
    volatile uint16_t tail;
} ringBuffer_t; 

typedef enum bufErrCodes_e
{
    ENQUEUE_OK,
    DEQUEUE_OK, 
    EMPTY, 
    FULL_OVERWRITE,
} buffErrCodes_e; 

void initRingBuffer(ringBuffer_t *buffer, uint8_t *storage, uint16_t size);

buffErrCodes_e enqueue(ringBuffer_t *buffer, uint8_t input);
buffErrCodes_e dequeue(ringBuffer_t *buffer, uint8_t *output); 

bool bufIsEmpty(ringBuffer_t *buffer); 

