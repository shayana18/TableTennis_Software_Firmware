#include "robot_fsm.h"

#include "motion_execute.h"
#include "robot_runtime.h"

void delta_fsm_bind_io(mailbox_t *mailbox, io_motor_com_t *motor_com)
{
  robot_runtime_bind(mailbox, motor_com);
}

void delta_fsm_on_timer_tick(void)
{
  motion_execute_on_timer_tick();
}

void delta_fsm_init(robot_t *robot)
{
  robot->state = STATE_OFF;
  robot->current_pos = robot_get_current_pos();
  robot->current_target.type = TARGET_NONE;
  robot->current_target.pos = robot->current_pos;
  robot->current_target.t_arrival_s = 0.0f;
  robot->current_move_plan.active = false;

  robot->flag_new_target = false;
  robot->flag_ready_to_move = false;
  robot->flag_path_done = false;
  robot->flag_path_abort = false;
  robot->flag_fault = false;
  robot->flag_pc_error = false;

  motion_execute_reset_scheduler();

  // debug prints
float sample_q1, sample_q2, sample_q3;
if (!robot_get_joint_angles(&sample_q1, &sample_q2, &sample_q3)) {
  robot_runtime_send_status("ERR: no init q\r\n");
} else {
  long q1_cdeg = (long)lroundf(sample_q1); 
  long q2_cdeg = (long)lroundf(sample_q2);
  long q3_cdeg = (long)lroundf(sample_q3);

  char msg[96];
  snprintf(msg, sizeof(msg), "INIT q(cdeg): %ld %ld %ld\r\n", q1_cdeg, q2_cdeg, q3_cdeg);
  robot_runtime_send_status(msg);
}


}

void delta_fsm(robot_t *robot)
{

  if (robot->flag_fault) {
    robot->state = STATE_FAULT;
    robot_runtime_send_status("FAULT HAS OCCURRED\r\n");
    return;
  }

  switch (robot->state) {
    case STATE_OFF:
      robot_runtime_send_status("STATE: OFF\r\n");
      robot->state = STATE_UNHOMED;
      robot_runtime_send_status("STATE: UNHOMED\r\n");
      robot_runtime_send_status("WAITING FOR HOME TARGET\r\n");
      break;

    case STATE_UNHOMED:
      if (robot_runtime_pop_target(&robot->current_target)) {
        // Maybe add function to see if limit switches are engaged
        if (robot->current_target.type == TARGET_HOME) {
          robot_runtime_send_status("HOMING...\r\n");
          motion_execute_make_home_target(robot);
          robot->state = STATE_PLAN;
          robot_runtime_send_status("STATE: PLAN\r\n");
        }
      }
      break;

    case STATE_PLAN:
      if (!robot_target_in_workspace(robot->current_target.pos)) {
        robot_runtime_send_status("TARGET IS INVALID\r\n");
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
      } else {
        motion_execute_plan(robot);
        motion_execute_start(robot);
        if (!robot->flag_ready_to_move) {
          robot->state = STATE_IDLE;
          set_idle(robot);
          robot_runtime_send_status("STATE: IDLE\r\n");
          break;
        }
        robot->state = STATE_MOVE;
        robot_runtime_send_status("STATE: MOVE\r\n");
      }
      break;

    case STATE_IDLE:
      if (robot_runtime_pop_target(&robot->current_target)) {
        robot->state = STATE_PLAN;
      }
      break;

    case STATE_MOVE:

    // Continuously update motion
      if (motion_execute_consume_tick_due()) {
        motion_execute_tick(robot);
      }

      if (robot->flag_path_abort) {
        robot_runtime_send_status("STOP MESSAGE RECEIVED\r\n");
        motion_execute_stop_all();
        robot->state = STATE_IDLE;
        robot_runtime_send_status("STATE: IDLE\r\n");
        break;
      }

      if (robot->flag_path_done) {
        if (robot->current_target.type == TARGET_INTERCEPT) {
          robot->state = STATE_STRIKE;
          robot_runtime_send_status("STATE: STRIKE\r\n");
        } else if (robot->current_target.type == TARGET_STRIKE) {
          motion_execute_make_home_target(robot);
          robot->state = STATE_PLAN;
          robot_runtime_send_status("STRIKE DONE -> HOME\r\n");
        } else if (robot->current_target.type == TARGET_HOME) {
          robot->state = STATE_IDLE;
          set_idle(robot);
          robot_runtime_send_status("REACHED HOME\r\n");
          robot_runtime_send_status("STATE: IDLE\r\n");
            
          // Current Joint Angle at Completion
          float sample_q1, sample_q2, sample_q3;
          if (!robot_get_joint_angles(&sample_q1, &sample_q2, &sample_q3)) {
            robot_runtime_send_status("ERR: no init q\r\n");
          } else {
            long q1_cdeg = (long)lroundf(sample_q1); // centi-deg
            long q2_cdeg = (long)lroundf(sample_q2);
            long q3_cdeg = (long)lroundf(sample_q3);

            char msg[96];
            snprintf(msg, sizeof(msg), "COMPLETED Q: %ld %ld %ld\r\n", q1_cdeg, q2_cdeg, q3_cdeg);
            robot_runtime_send_status(msg);
          }
        } else {
          robot->state = STATE_IDLE;
          set_idle(robot);
          robot_runtime_send_status("STATE: IDLE\r\n");
        }
      }
      break;

    case STATE_STRIKE:
      motion_execute_plan_strike(robot);
      if (!robot_target_in_workspace(robot->current_target.pos)) {
        robot_runtime_send_status("PATH_INVALID, SENDING HOME\r\n");
        motion_execute_make_home_target(robot);
        robot->state = STATE_PLAN;
      } else {
        motion_execute_plan(robot);
        motion_execute_start(robot);
        if (!robot->flag_ready_to_move) {
          robot->state = STATE_IDLE;
          set_idle(robot);
          robot_runtime_send_status("STATE: IDLE\r\n");
          break;
        }
        robot->state = STATE_MOVE;
        robot_runtime_send_status("STATE: MOVE\r\n");
      }
      break;

    case STATE_FAULT:
    default:
      safety_enter_fault_mode();
      motion_execute_stop_all();
      robot_runtime_send_status("FAULT\r\n");
      break;
  }
}
