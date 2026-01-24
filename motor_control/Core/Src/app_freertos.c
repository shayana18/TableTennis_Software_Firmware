/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : app_freertos.c
  * Description        : FreeRTOS applicative file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_freertos.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
typedef StaticTask_t osStaticThreadDef_t;
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

/* USER CODE END Variables */
/* Definitions for motor_coms */
osThreadId_t motor_comsHandle;
uint32_t MyBufferTask01[ 128 ];
osStaticThreadDef_t MycontrolBlocTask01;
const osThreadAttr_t motor_coms_attributes = {
  .name = "motor_coms",
  .stack_mem = &MyBufferTask01[0],
  .stack_size = sizeof(MyBufferTask01),
  .cb_mem = &MycontrolBlocTask01,
  .cb_size = sizeof(MycontrolBlocTask01),
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for control_task */
osThreadId_t control_taskHandle;
uint32_t MyBufferTask02[ 128 ];
osStaticThreadDef_t MycontrolBlocTask02;
const osThreadAttr_t control_task_attributes = {
  .name = "control_task",
  .stack_mem = &MyBufferTask02[0],
  .stack_size = sizeof(MyBufferTask02),
  .cb_mem = &MycontrolBlocTask02,
  .cb_size = sizeof(MycontrolBlocTask02),
  .priority = (osPriority_t) osPriorityNormal2,
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */
  /* creation of motor_coms */
  motor_comsHandle = osThreadNew(StartDefaultTask, NULL, &motor_coms_attributes);

  /* creation of control_task */
  control_taskHandle = osThreadNew(StartTask02, NULL, &control_task_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

}
/* USER CODE BEGIN Header_StartDefaultTask */
/**
* @brief Function implementing the motor_coms thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN motor_coms */
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END motor_coms */
}

/* USER CODE BEGIN Header_StartTask02 */
/**
* @brief Function implementing the control_task thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartTask02 */
void StartTask02(void *argument)
{
  /* USER CODE BEGIN control_task */
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END control_task */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

