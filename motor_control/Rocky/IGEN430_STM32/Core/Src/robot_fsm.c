/*
 * fsm.c
 *
 *  Created on: Feb 3, 2026
 *      Author: rocky
 */

#include "robot_fsm.h"
#include "services/service_motion.h"
#include "services/service_comms.h"
#include "services/service_safety.h"
#include "robot.h"



// ==========================================================



void delta_fsm_init(robot_t *robot)
{
  robot->state = STATE_OFF;
  robot->current_pos = robot_get_current_pos(); 			// DRIVER FUNCTION
  robot->current_target.type = TARGET_NONE;

  robot->flag_new_target 	= false;
  robot->flag_ready_to_move = false;
  robot->flag_path_done  	= false;
  robot->flag_path_abort 	= false;
  robot->flag_fault      	= false;
  robot->flag_pc_error   	= false;
}

void delta_fsm(robot_t *robot)
{
  // Process any incoming PC messages
  comms_process_data();

  //

	if (robot->flag_fault) {
	  robot->state = STATE_FAULT;
	  comms_send_status("FAULT HAS OCCURRED!!\n");
	  return;
	}

  switch (robot->state)
  {
    case STATE_OFF:
    {
	//
		comms_send_status("STATE: OFF!\n");
		comms_send_status("Powered ON!\n");
		robot->state = STATE_UNHOMED;
		comms_send_status("STATE: UNHOMED!\n");

    } break;

    case STATE_UNHOMED:
	{
		// Check if new target arrived
		if (comms_pop_new_target(&robot->current_target)) {

			if (!safety_check_joint_limits()) {
				comms_send_status("ROBOT JOINTS ARE IN INVALID STATE\n");
			}else if (robot->current_target.type == TARGET_HOME) {
				motion_make_home_target(&robot);
				robot->state = STATE_PLAN;
				comms_send_status("STATE: PLAN!\n");
			}
			break;

		}
		comms_send_status("PLEASE HOME THE ROBOT FIRST!\n");
		// Must home before continuing - will check if robot is even in workspace upon power cycle

	} break;

    case STATE_PLAN:
	{


		// Plan trajectory for the current target
		bool path_ok = safety_check_target(robot->current_target.pos);


		if (!path_ok) {
		comms_send_status("PATH_INVALID\n");
		robot->state = STATE_IDLE;
		set_idle(robot);
		comms_send_status("STATE: IDLE\n");
		} else {

		motion_plan(&robot);
		motion_start(&robot);
		robot->state = STATE_MOVE;
		comms_send_status("STATE: MOVE\n");
		}
	} break;

    case STATE_IDLE:
    {
      // Check if new target arrived
		if (comms_pop_new_target(&robot->current_target)) {
			robot->state = STATE_PLAN;
		}

    } break;


    case STATE_MOVE:
    {

      // If ISR aborted (speed limit / singularity / comm timeout)
      if (robot->flag_path_abort) {
        comms_send_status("STOP MESSAGE RECIEVED\n");
        stop_motion();
        robot->state = STATE_IDLE;
        comms_send_status("STATE: IDLE\n");
        break;
      }

      // When plan completes
      if (robot->flag_path_done) {
        // If we just reached intercept target, optionally do “strike”
        if (robot->current_target.type == TARGET_INTERCEPT) {
          robot->state = STATE_STRIKE;
          comms_send_status("STATE: STRIKE\n");
          break;
        } else if (robot->current_target.type == TARGET_HOME) {

          robot->state = STATE_IDLE;
          comms_send_status("REACHED HOME\n");
          set_idle(robot);
          comms_send_status("STATE: IDLE\n");
          break;
        }

      }

    } break;

    case STATE_STRIKE:
    {


    	motion_plan_strike(&robot);
		// Plan trajectory for the strike target
		bool path_ok = safety_check_target(robot->current_target.pos);

		if (!path_ok) {
		comms_send_status("PATH_INVALID, Sending to Home pos\n");
		motion_make_home_target(&robot);
        robot->state = STATE_PLAN;
        break;
		}

		motion_plan(&robot);
		motion_strike(&robot);

		if (robot->flag_path_done) {
			motion_make_home_target(&robot);
	    robot->state = STATE_PLAN;
		}
    } break;

    case STATE_FAULT:
    default:
    {
      // Enter safe mode
      safety_enter_fault_mode();
      stop_motion();
      comms_send_status("FAULT\n");

      // MVP: stay here until power cycle / manual clear
      // If you want automatic recovery, do:
      // if (!safety_any_fault_active()) { ctx->flag_fault=false; ctx->state=STATE_IDLE; set_idle(ctx); }
    } break;
  }
}


