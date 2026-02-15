#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "shared_types.h"

typedef struct mailbox
{
    float data[TARGET_MSG_FLOAT_COUNT];
    volatile bool new_interception_point_in;
} mailbox_t;

void mailbox_init(mailbox_t *mb);

// Called from ISR: overwrite with newest data and set flag.
// 'data' must point to TARGET_MSG_FLOAT_COUNT floats.
void mailbox_mail_arrived(mailbox_t *mb, const float *data);

// Called from main loop/thread: returns true if new mail was consumed.
// Resets flag, zeros mailbox data, and parses into target_t.
bool mailbox_mail_received(mailbox_t *mb, target_t *out_target);
