#include "ringBuffer.h"
#include <string.h>

static bool bufIsFull(ringBuffer_t *buffer);

void initRingBuffer(ringBuffer_t *buffer)
{
    buffer->head = 0;
    buffer->tail = 0;
}

bool bufIsEmpty(ringBuffer_t *buffer)
{
    return buffer->head == buffer->tail;
}

buffErrCodes_e enqueue(ringBuffer_t *buffer, const float *input)
{

    if (bufIsFull(buffer)) {
        buffer->head = (uint8_t)((buffer->head + 1) % MAX_BUF_SIZE); // drop oldest as it will get overwritten
        memcpy(buffer->buf[buffer->tail], input, sizeof(buffer->buf[buffer->tail]));
        buffer->tail = (uint8_t)((buffer->tail + 1) % MAX_BUF_SIZE);
        return FULL_OVERWRITE;
    }

    memcpy(buffer->buf[buffer->tail], input, sizeof(buffer->buf[buffer->tail]));
    buffer->tail = (uint8_t)((buffer->tail + 1) % MAX_BUF_SIZE);
    return ENQUEUE_OK;
}

buffErrCodes_e dequeue(ringBuffer_t *buffer, float *output)
{
    if (bufIsEmpty(buffer)) {
        return EMPTY;
    }
    memcpy(output, buffer->buf[buffer->head], sizeof(buffer->buf[buffer->head]));
    buffer->head = (uint8_t)((buffer->head + 1) % MAX_BUF_SIZE);
    return DEQUEUE_OK;
}

static bool bufIsFull(ringBuffer_t *buffer)
{
    return ((uint8_t)((buffer->tail + 1) % MAX_BUF_SIZE)) == buffer->head;
}
