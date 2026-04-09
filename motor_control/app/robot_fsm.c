#include "robot_fsm.h"

#include "motion_execute.h"
#include "robot.h"
#include "robot_runtime.h"
#include "shared_types.h"
#include "stm32c0xx_hal.h"

#include <math.h>
#include <stdio.h>


int x = 0;

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

  // Print Current Joint Angle at Home Completion [ DO NOT DELETE - USEFUL FOR DEBUGGING ]
  robot_runtime_send_status("new Program 1");
  print_joint_angles();
}

void delta_fsm_bind_io(mailbox_t *mailbox, io_motor_com_t *motor_com)
{
  robot_runtime_bind(mailbox, motor_com);
}

void delta_fsm_on_timer_tick(void)
{
  motion_execute_on_timer_tick();
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
    
    while (x < 1){
      robot_runtime_send_status("STATE: OFF\r\n");
      x++;
    }
      

      if(robot_runtime_pop_target(&robot->current_target) && robot->current_target.type == TARGET_HOME)
      {
          robot_runtime_send_status("HOMING...\r\n");
          motion_execute_make_home_target(robot);
          robot->state = STATE_PLAN;
          robot_runtime_send_status("STATE: PLAN\r\n");
      }
      break;
      
    case STATE_IDLE:
      if (!robot_runtime_pop_target(&robot->current_target)) {
        break;
      }

      {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "RX TARGET: type=%d x=%ld y=%ld z=%ld\r\n",
                 (int)robot->current_target.type,
                 (long)lroundf(robot->current_target.pos.x),
                 (long)lroundf(robot->current_target.pos.y),
                 (long)lroundf(robot->current_target.pos.z));
        robot_runtime_send_status(msg);
      }

      if(robot->current_target.type == TARGET_INTERCEPT || robot->current_target.type == TARGET_TEST)
      {
        robot->state = STATE_PLAN;
        robot_runtime_send_status("STATE: PLAN\r\n");
      }
      else if(robot->current_target.type == TARGET_HOME)
      {
          robot_runtime_send_status("HOMING...\r\n");
          motion_execute_make_home_target(robot);
          robot->state = STATE_PLAN;
          robot_runtime_send_status("STATE: PLAN\r\n");
        }
      break;


    case STATE_PLAN:

      motion_execute_plan(robot);

      // Planning may intentionally conclude with no motion needed
      // (e.g., target is already within 1 mm of current pose).
      // Treat that as success, not a planning error.
      if (robot->flag_path_done) {
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
        break;
      }

      // If planning already aborted itself (workspace/IK/etc), do not run start.
      if (robot->flag_path_abort) {
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
        break;
      }

      motion_execute_start(robot);
      if (!robot->flag_ready_to_move) {
        robot_runtime_send_status("Planning Failed\r\n");
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
        break;
      }
      robot->state = STATE_MOVE;
      robot_runtime_send_status("STATE: MOVE\r\n");
      
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
          HAL_Delay(200);
          motion_execute_make_home_target(robot);
          robot->state = STATE_PLAN;
          // robot->state = STATE_IDLE;
          // set_idle(robot);
          
          robot_runtime_send_status("STRIKE DONE -> HOME\r\n");
        } else if (robot->current_target.type == TARGET_HOME) {
          robot->state = STATE_IDLE;
          set_idle(robot);
          robot_runtime_send_status("REACHED HOME\r\n");
          robot_runtime_send_status("STATE: IDLE\r\n");
        } else { // Test target type
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
        print_joint_angles();
        }

      }
      break;

    case STATE_STRIKE:
      motion_execute_prepare_strike(robot);
      if (!robot_EE_in_workspace(robot->current_target.pos)) {
        robot->state = STATE_IDLE;
        set_idle(robot);
        robot_runtime_send_status("STATE: IDLE\r\n");
      } else {
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
