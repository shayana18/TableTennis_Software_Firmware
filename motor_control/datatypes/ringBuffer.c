#include "ringBuffer.h"

static bool bufIsFull(ringBuffer_t *buffer);

void initRingBuffer(ringBuffer_t *buffer, uint8_t *storage, uint16_t size)
{
    buffer->buf = storage;
    buffer->size = size;
    buffer->head = 0;
    buffer->tail = 0;
}

bool bufIsEmpty(ringBuffer_t *buffer)
{
    return buffer->head == buffer->tail;
}

buffErrCodes_e enqueue(ringBuffer_t *buffer, uint8_t input)
{
    if (bufIsFull(buffer)) {
        buffer->head = (uint16_t)((buffer->head + 1) % buffer->size); // drop oldest as it will get overwritten
        buffer->buf[buffer->tail] = input;
        buffer->tail = (uint16_t)((buffer->tail + 1) % buffer->size);
        return FULL_OVERWRITE;
    }

    buffer->buf[buffer->tail] = input;
    buffer->tail = (uint16_t)((buffer->tail + 1) % buffer->size);
    return ENQUEUE_OK;
}

buffErrCodes_e dequeue(ringBuffer_t *buffer, uint8_t *output)
{
    if (bufIsEmpty(buffer)) {
        return EMPTY;
    }
    *output = buffer->buf[buffer->head];
    buffer->head = (uint16_t)((buffer->head + 1) % buffer->size);
    return DEQUEUE_OK;
}

static bool bufIsFull(ringBuffer_t *buffer)
{
    return ((uint16_t)((buffer->tail + 1) % buffer->size)) == buffer->head;
}
