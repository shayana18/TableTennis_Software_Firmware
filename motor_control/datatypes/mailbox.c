#include "mailbox.h"
#include <string.h>

static void mailbox_parse_target(const float *data, target_t *out_target)
{
    out_target->type = (target_type)((int)data[TARGET_MSG_TYPE]);
    out_target->intercept_pos.x = data[TARGET_MSG_X_POS];
    out_target->intercept_pos.y = data[TARGET_MSG_Y_POS];
    out_target->intercept_pos.z = data[TARGET_MSG_Z_POS];
    out_target->intercept_vel.x = data[TARGET_MSG_X_VEL];
    out_target->intercept_vel.y = data[TARGET_MSG_Y_VEL];
    out_target->intercept_vel.z = data[TARGET_MSG_Z_VEL];
    out_target->intercept_time = data[TARGET_MSG_INTERCEPT_TIME];
    out_target->time_sent = data[TARGET_MSG_TIME_SENT];
    out_target->timestamp = data[TARGET_MSG_TIMESTAMP];
}

void mailbox_init(mailbox_t *mb)
{
    mb->new_interception_point_in = false;
    memset(mb->data, 0, sizeof(mb->data));
}

void mailbox_mail_arrived(mailbox_t *mb, const float *data)
{
    memcpy(mb->data, data, sizeof(mb->data));
    mb->new_interception_point_in = true;
}

bool mailbox_mail_received(mailbox_t *mb, target_t *out_target)
{
    float local[TARGET_MSG_FLOAT_COUNT];

    if(!mb->new_interception_point_in)
    {
        return false; 
    }

    mb->new_interception_point_in = false;
    memcpy(local, mb->data, sizeof(mb->data));

    mailbox_parse_target(local,out_target);

    memset(mb->data, 0, sizeof(mb->data));
    return true;
}

void mailbox_clear(mailbox_t *mb)
{
    mb->new_interception_point_in = false;
    memset(mb->data, 0, sizeof(mb->data));
}
