/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include "main.h"
#include <ctype.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// Functions sent by host
#define Set_Origin              0x00
#define Go_Absolute_Pos         0x01
#define Go_Relative_Pos         0x03
#define General_Read            0x0e
#define Set_MainGain            0x10
#define Set_SpeedGain           0x11
#define Set_IntGain             0x12
#define Set_HighSpeed           0x14
#define Set_HighAccel           0x15
#define Set_Pos_OnRange         0x16
#define Set_GearNumber          0x17
#define Read_MainGain           0x18
#define Read_HighSpeed			0x1c


// Functions sent by DYN drive
#define Is_MainGain             0x10
#define Is_SpeedGain            0x11
#define Is_IntGain              0x12
#define Is_TrqCons              0x13
#define Is_HighSpeed            0x14
#define Is_HighAccel            0x15
#define Is_Driver_ID            0x16
#define Is_Pos_OnRange          0x17
#define Is_Status               0x19
#define Is_Config               0x1a
#define Is_AbsPos32             0x1b
#define Is_TrqCurrent           0x1e

#define UART2_BAUD                115200 //Set Baud Rate for USB Communication (To Tablet or Computer)
#define UART3_BAUD                38400
//USE PIN PC4 ON CN10 FOR TX

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM3_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM1_Init(void);
/* USER CODE BEGIN PFP */
void clockwiseMotion();
void counterClockwiseMotion();
void speedControl(char*);
void absoluteMove(char*);
void absMoveSetup();
int goHome();

//Reading & Display Function Prototypes
void sendString();
void classifyReadString(long, char, char*);
char* ltoa(long, char*, int);

//Timer Functions
void timeDelay(int);

//Drive Function Prototypes
// void Drive_Reset_RS232();
// void Drive_Enable_RS232();
// void Drive_Disable_RS232();
void move_rel32(char,long);
void ReadMotorTorqueCurrent(char);
void ReadMotorPosition32(char);
void ReadMotorSpeed32(char);
void move_abs32(char,long);
void Turn_const_speed(char,long);
void ReadPackage();
void Get_Function();
int32_t Cal_SignValue(unsigned char*);
long Cal_Value(unsigned char*);
void Send_Package(char,long);
void Make_CRC_Send(unsigned char, unsigned char*);

//Menu Functions
void speedMenu();
void positionMenu();
char updateMotorData(char,int,long);
// Shayan Functions
void setMotorSpeed(char motorID);
void setAllMotorsSpeed(void);
void indiv_motor_driv(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
char InputBuffer[256];                          //Input buffer from RS232,
char OutputBuffer[256]; //Output buffer to RS232,
char outputString[20];
unsigned char InBfTopPointer, InBfBtmPointer;   //input buffer pointers
unsigned char OutBfTopPointer, OutBfBtmPointer; //output buffer pointers
unsigned char Read_Package_Buffer[8], Read_Num, Read_Package_Length, Global_Func;
unsigned char MotorPosition32Ready_Flag, MotorTorqueCurrentReady_Flag, MainGainRead_Flag, MotorSpeed32Ready_Flag;
unsigned char Driver_MainGain, Driver_SpeedGain, Driver_IntGain, Driver_TrqCons, Driver_HighSpeed, Driver_HighAccel,Driver_ReadID,Driver_Status,Driver_Config,Driver_OnRange;
long Motor_Pos32, MotorTorqueCurrent, Motor_Speed32;
int speed = 0;
int position = 0;
int sendSpeed;
char speedParse;
int motorPower;
char motorDirection = 0; //motorDirection = 1 (CW), motorDirection = 2 (CCW), 0 on startup
char delayDone = 0;

char posMenu = 0;
char trqMenu = 0;


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM3_Init();
  MX_USART2_UART_Init();
  MX_USART1_UART_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */
  motorPower = 1;
  char rx_data; // Variable to store received character
  int rx_index = 0; // Index to keep track of received characters
  char sendMsg[256];
  char commandType;
  char commandValue;
  HAL_StatusTypeDef status;
  char menu[] = "\x1b[1m\r\nDaisy Chain Motor Test:\r\n\x1b[0m\r\n \x1b[1;36m[1]\tDrive a specific motor\r\n\x1b[0m \x1b[1;36m[2]\tDrive All Motors\r\n\x1b[0m\r\nSelect an option: ";

  Turn_const_speed(1,0);
  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED_GREEN);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */
    HAL_UART_Transmit(&huart2, (uint8_t *)menu, strlen(menu), HAL_MAX_DELAY);
    HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY);

    switch (rx_data)
    {
      case '1':
        indiv_motor_driv();
        break;
      case '2':
        setAllMotorsSpeed();
        break;
      case '3':
        HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1;31mAll motors turned OFF\r\n\x1b[0m", strlen("\r\n\x1b[1;31mAll motors turned OFF\r\n\x1b[0m"), HAL_MAX_DELAY);
        Turn_const_speed(1, 0);
        Turn_const_speed(2, 0);
        Turn_const_speed(3, 0);
        HAL_Delay(1500);
        break;
      default:
        HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1;31mInvalid selection!\r\n\x1b[0m", strlen("\r\n\x1b[1;31mInvalid selection!\r\n\x1b[0m"), HAL_MAX_DELAY);
        HAL_Delay(500);
        break;
    }
  }
  /* USER CODE END 3 */

}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSE;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 7;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 49999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 600;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.BreakAFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.Break2AFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_OC_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_TIMING;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_OC_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 38400;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PA5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void indiv_motor_driv(void)
{
  char ind_drive_men[] = "\x1b[1m\r\nIndividual Motor Control:\r\n\x1b[0m\r\n \x1b[1;36m[1]\tDrive Motor 1\r\n\x1b[0m \x1b[1;36m[2]\tDrive Motor 2\r\n\x1b[0m \x1b[1;36m[3]\tDrive Motor 3\r\n\x1b[0m \x1b[1;36m[4]\tReturn to Main Menu\r\n\x1b[0m\r\nSelect an option: ";
  char rx_data;
  char exit_menu = 0;

  while(exit_menu == 0)
  {
    HAL_UART_Transmit(&huart2, (uint8_t *)ind_drive_men, strlen(ind_drive_men), HAL_MAX_DELAY);
    HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY);

    switch(rx_data)
    {
      case '1':
        setMotorSpeed(2);
        HAL_Delay(500);
        break;
      case '2':
        setMotorSpeed(3);
        HAL_Delay(500);
        break;
      case '3':
        setMotorSpeed(4);
        HAL_Delay(500);
        break;
      case '4':
        HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1mReturning to Main Menu...\r\n\x1b[0m", strlen("\r\n\x1b[1mReturning to Main Menu...\r\n\x1b[0m"), HAL_MAX_DELAY);
        HAL_Delay(500);
        exit_menu = 1;
        break;
      default:
        HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1;31mInvalid selection!\r\n\x1b[0m", strlen("\r\n\x1b[1;31mInvalid selection!\r\n\x1b[0m"), HAL_MAX_DELAY);
        HAL_Delay(500);
        break;
    }
  }
}

/* Function Name: setAllMotorsSpeed
 * Author: Shayan Ajmal 
 * Date: 2026-02-03
 * Purpose: Helper function to set speed for all three motors simultaneously. Handles user input validation
 * and sends the speed command to all motors
 * */
void setAllMotorsSpeed(void){
  char rx_data;
  int rx_index = 0;
  char rx_buffer[30];
  char sendMsg[100];
  char isTyping = 1;
  int speedTemp = 0;

  HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1mEnter Speed (RPM) for All Motors: \x1b[0m", strlen("\r\n\x1b[1mEnter Speed (RPM) for All Motors: \x1b[0m"), HAL_MAX_DELAY);

  while(isTyping == 1){

    if (HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY) == HAL_OK){

      if(rx_data == '\r'){

        rx_buffer[rx_index] = '\0';
        rx_index = 0;
        speedTemp = atoi(rx_buffer);
        //insert software limits on speed here
        if(speedTemp < 0 || speedTemp > 3000){
          HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m", strlen("\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m"), HAL_MAX_DELAY);
          memset(rx_buffer,' ',30);
          HAL_Delay(250);
          HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\x1b[1mEnter Speed (RPM) for All Motors: \x1b[0m", strlen("\r\n\x1b[1mEnter Speed (RPM) for All Motors: \x1b[0m"), HAL_MAX_DELAY);
        }
        else{
          isTyping = 0;
          speed = speedTemp;
          sprintf(sendMsg,"\r\n\n\x1b[1mAll motors speed set to %d RPM\x1b[0m\r\n", speedTemp);
          HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY);
          Turn_const_speed(2, speed);
          Turn_const_speed(3, speed);
          Turn_const_speed(4, speed);
          HAL_Delay(1000);
        }

      }
      else{

        rx_buffer[rx_index++] = rx_data;

        if (rx_index >= 30){
          rx_index = 0;
        }

        HAL_UART_Transmit(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY);

      }

    }

  }
}

void speedMenu(){

	char speedMenu[] = "\x1b[1m\r\nSpeed Control:\r\n\x1b[0m\r\n \x1b[1;36m[1]\tSet Direction\r\n\x1b[0m \x1b[1;36m[2]\tSet Speed\r\n\x1b[0m \x1b[1;36m[3]\tGO!\r\n\x1b[0m\r\nPress [X] at any point to exit speed control.\r\n\r\n";
	char rx_data; // Variable to store received character
	int rx_index = 0;
	char rx_buffer[30]; // Buffer to store received string
	char sendMsg [100];
	char spdMenu = 1;
	char isTyping = 0;
	char dirSelect = 0;
	int speedTemp = 0;

	while(spdMenu == 1){

		HAL_UART_Transmit(&huart2, (uint8_t *)speedMenu, strlen(speedMenu), HAL_MAX_DELAY); // Transmit menu over UART

		HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Receive user input

		switch(rx_data){
		case '1':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\r\nChoose the direction of rotation:\r\n\r\n [1]\tClockwise\r\n [2]\tCounter-Clockwise\r\n\r\n\r\n", strlen("\r\nChoose the direction of rotation:\r\n\r\n [1]\tClockwise\r\n[2]\tCounter-Clockwise\r\n\r\n\r\n"), HAL_MAX_DELAY);
			HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Receive user input
			dirSelect = 1;

			while(dirSelect == 1){
				switch(rx_data){
				case '1':
					HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1mClockwise\r\n\x1b[0m", strlen("\x1b[1mClockwise\r\n\x1b[0m"), HAL_MAX_DELAY);
					motorDirection = 1;
					dirSelect = 0;
					break;
				case '2':
					HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1mCounter-Clockwise\r\n\x1b[0m", strlen("\x1b[1mCounter-Clockwise\r\n\x1b[0m"), HAL_MAX_DELAY);
					motorDirection = 2;
					dirSelect = 0;
					break;
				default:
					HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1;31mInvalid selection!\r\n\x1b[0m", strlen("\x1b[1;31mInvalid selection!\r\n\x1b[0m"), HAL_MAX_DELAY);
					break;
				}
			}

			break;
		case '2':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\r\nEnter Speed (RPM): ", strlen("\r\nEnter Speed (RPM): "), HAL_MAX_DELAY);
			isTyping = 1;
			rx_index = 0;

			while(isTyping == 1){

				if (HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY) == HAL_OK){


					if(rx_data == 'x' || rx_data =='X'){
						isTyping = 0;
						spdMenu = 0;
						HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
						HAL_Delay(250);
					}

					if(rx_data == '\r'){

						rx_buffer[rx_index] = '\0';
						rx_index = 0;
						speedTemp = atoi(rx_buffer);
						//insert software limits on speed here
						if(speedTemp < 0 || speedTemp > 3000){
							HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m", strlen("\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m"), HAL_MAX_DELAY);
							memset(rx_buffer,' ',30);
							HAL_Delay(250);
							HAL_UART_Transmit(&huart2, (uint8_t *)"Enter Speed (RPM): ", strlen("Enter Speed (RPM): "), HAL_MAX_DELAY);
						}
						else{
							isTyping = 0;
							speed = speedTemp;
							sprintf(sendMsg,"\r\n\n\x1b[1mMotor speed set to %d RPM\x1b[0m\r\n", speedTemp);
							HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
						}


					}
					else{

						rx_buffer[rx_index++] = rx_data;

		                if (rx_index >= 30){
		                    // Handle buffer overflow (e.g., by discarding excess characters)
		                    rx_index = 0; // Reset buffer index
		                }

		                HAL_UART_Transmit(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Transmit menu over UART

					}

				}

			}
			break;
		case '3':
			if(motorDirection == 1){
				HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
				HAL_Delay(250);
				sprintf(sendMsg,"\x1b[1mMoving Clockwise at %d RPM\r\n\r\n[Space] - Change Direction\t[X] - Main Menu\t\t[B] - Previous Menu\x1b[0m\r\n\n", speedTemp);
				HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
				clockwiseMotion();
				char code = updateMotorData(2, speedTemp,0);
				if(code == 1){
					spdMenu = 0;
				}

				//scan for user inputs, format speed and position output
			}
			else if(motorDirection == 2){
				HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
				HAL_Delay(250);
				sprintf(sendMsg,"\x1b[1mMoving Counter-Clockwise at %d RPM\r\n\r\n[Space] - Change Direction\t[X] - Main Menu\t\t[B] - Previous Menu\x1b[0m\r\n\n", speedTemp);
				HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
				counterClockwiseMotion();
				char code = updateMotorData(2, speedTemp,0);
				if(code == 1){
					spdMenu = 0;
				}
				//scan for user inputs, format speed and position output
			}
			else{
				HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1;31mMotor direction not set!\r\n\r\nSet motor direction prior to moving!\n\r\x1b[0m", strlen("\x1b[1;31mMotor direction not set!\r\n\r\nSet motor direction prior to moving!\n\r\x1b[0m"), HAL_MAX_DELAY);
			}
			break;
		case 'X':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
			HAL_Delay(250);
			spdMenu = 0;
			break;
		case 'x':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
			HAL_Delay(250);
			spdMenu = 0;
			break;
		default:
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1;31mInvalid selection!\r\n\x1b[0m", strlen("\x1b[1;31mInvalid selection!\r\n\x1b[0m"), HAL_MAX_DELAY);
			break;
		}
	}

}

void positionMenu(){

	char positionMenu[] = "\x1b[1m\r\nPosition Control:\r\n\x1b[0m\r\n \x1b[1;36m[1]\tAbsolute Move (in pulses)\r\n\x1b[0m \x1b[1;36m[2]\tRelative Move (by revolution count)\r\n\x1b[0m\r\nPress [X] at any point to exit position control.\r\n\r\n";
	char rx_data; // Variable to store received character
	int rx_index = 0;
	char currentPos[256];
	char sendMsg [100];
	char posMenu = 1;
	char isTyping = 0;
	long posTemp = 0;
	int revCnt = 0;
	char rx_buffer[30]; // Buffer to store received string

	while(posMenu == 1){

		HAL_UART_Transmit(&huart2, (uint8_t *)positionMenu, strlen(positionMenu), HAL_MAX_DELAY); // Transmit menu over UART
		HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Receive user input

		switch(rx_data){
		case '1':
			// ReadMotorPosition32(1);
			HAL_Delay(25);
			sprintf(currentPos,"\x1b[1m\r\nCurrent Motor Position:\x1b[0m \x1b[1;36m %ld [Pulses]\x1b[0m\n\n",Motor_Pos32);
			HAL_UART_Transmit(&huart2, (uint8_t *)currentPos, strlen(currentPos), HAL_MAX_DELAY); // Transmit menu over UART
			HAL_UART_Transmit(&huart2, (uint8_t *)"\r\nEnter the Destination Position (Encoder Pulses): ", strlen("\r\nEnter the Destination Position (Encoder Pulses): "), HAL_MAX_DELAY);
			isTyping = 1;
			rx_index = 0;

			while(isTyping == 1){

				if (HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY) == HAL_OK){


					if(rx_data == 'x' || rx_data =='X'){
						isTyping = 0;
						posMenu = 0;
						HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
						HAL_Delay(250);
					}

					if(rx_data == '\r'){
						rx_buffer[rx_index] = '\0';
						rx_index = 0;
						posTemp = strtol(rx_buffer,NULL,10);
						//insert software limits on speed here
						if(posTemp > 134217727 || posTemp < -134217728){
							HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nPosition must be between -134217728 and 134217727 Pulses!\n\n\r\x1b[0m", strlen("\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nPosition must be between -134217728 and 134217727 Pulses!\n\n\r\x1b[0m"), HAL_MAX_DELAY);
							memset(rx_buffer,' ',30);
							HAL_Delay(250);
							HAL_UART_Transmit(&huart2, (uint8_t *)"Enter the Destination Position (Encoder Pulses): ", strlen("Enter the Destination Position (Encoder Pulses): "), HAL_MAX_DELAY);
						}
						else{
							isTyping = 0;
							position = posTemp;
							HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
							HAL_Delay(25);
							sprintf(sendMsg,"\r\x1b[1mTarget Position set to %ld Pulses\x1b[0m\r\n\nPress [SPACE] to perform an Absolute Move.\r\n\n[B] to return to position menu\n\n\r[X] to return to main menu\r\n\n", posTemp);
							HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UARTbbbbb
							char code = updateMotorData(3,0,posTemp);
							if(code == 1){
								posMenu = 0;
							}

						}
					}
					if(isTyping == 1){

						rx_buffer[rx_index++] = rx_data;

		                if (rx_index >= 30){
		                    // Handle buffer overflow (e.g., by discarding excess characters)
		                    rx_index = 0; // Reset buffer index
		                }
						HAL_UART_Transmit(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Transmit menu over UART
					}
				}
			}
			break;
		case '2':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\r\nEnter desired number of rotations (Whole Numbers ONLY): ", strlen("\r\nEnter desired number of rotations (Whole Numbers Only): "), HAL_MAX_DELAY);
			isTyping = 1;

			while(isTyping == 1){

				if (HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY) == HAL_OK){


					if(rx_data == 'x' || rx_data =='X'){
						isTyping = 0;
						posMenu = 0;
						HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
						HAL_Delay(250);
					}

					if(rx_data == '\r'){
						rx_buffer[rx_index] = '\0';
						rx_index = 0;
						revCnt = atoi(rx_buffer);
						isTyping = 0;
						HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
						HAL_Delay(25);
						sprintf(sendMsg,"\r\x1b[1mTarget number of Revolutions set to: %d\x1b[0m\r\n\nPress [SPACE] to perform a Relative Move.\r\n\n[B] to return to position menu\n\n\r[X] to return to main menu\r\n\n",revCnt);
						HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
						char code = updateMotorData(4,revCnt,0);
						if(code == 1){
							posMenu = 0;
						}

					}
					if(isTyping == 1){

						rx_buffer[rx_index++] = rx_data;

		                if (rx_index >= 30){
		                    // Handle buffer overflow (e.g., by discarding excess characters)
		                    rx_index = 0; // Reset buffer index
		                }
						HAL_UART_Transmit(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY); // Transmit menu over UART
					}
				}
			}
			break;
		case 'X':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
			HAL_Delay(250);
			posMenu = 0;
			break;
		case 'x':
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
			HAL_Delay(250);
			posMenu = 0;
			break;
		default:
			HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[1;31mInvalid selection!\r\n\x1b[0m", strlen("\x1b[1;31mInvalid selection!\r\n\x1b[0m"), HAL_MAX_DELAY);
			break;
		}
	}

}

char updateMotorData(char controlMode, int RPM, long position){

	char reading = 1;
	char data[256];
	char sendMsg[100];
	char rx_data;
	char code = 0;

	HAL_NVIC_SetPriority(USART2_IRQn,0,0);
	HAL_NVIC_EnableIRQ(USART2_IRQn);

	while(reading == 1){

		ReadMotorPosition32(1);
		HAL_Delay(25);
		// ReadMotorSpeed32(1);
		HAL_Delay(25);
		sprintf(data,"\x1b[G\x1b[2KPostion: %ld [Pulses]\t||\tSpeed: %ld [RPM]",Motor_Pos32, Motor_Speed32);

		HAL_UART_Transmit(&huart2, (uint8_t *)data, strlen(data), HAL_MAX_DELAY); // Transmit menu over UART


		if(__HAL_UART_GET_FLAG(&huart2, UART_FLAG_RXNE) != RESET){
			HAL_UART_Receive(&huart2, &rx_data, 1, HAL_MAX_DELAY);

			if(rx_data == 'x' || rx_data == 'X'){
				reading = 0;
				code = 1;
				HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
				Turn_const_speed(1,0);
			}
			else if(rx_data == 'b' || rx_data == 'B'){
				reading = 0;
				Turn_const_speed(1,0);
				HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
				HAL_Delay(250);
			}
			else if(rx_data == ' ' && controlMode == 2){
				if (motorDirection == 1){
					motorDirection = 2;
					HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
					HAL_Delay(250);
					sprintf(sendMsg,"\x1b[1mMoving Counter-Clockwise at %d RPM\r\n\r\n[Space] - Change Direction\t[X] - Main Menu\t\t[B] - Previous Menu\x1b[0m\r\n\n", RPM);
					counterClockwiseMotion();
					HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
				}
				else if(motorDirection == 2){
					motorDirection = 1;
					HAL_UART_Transmit(&huart2, (uint8_t *)"\x1b[0;0H\x1b[2J", strlen("\x1b[0;0H\x1b[2J"), HAL_MAX_DELAY);
					HAL_Delay(250);
					sprintf(sendMsg,"\x1b[1mMoving Clockwise at %d RPM\r\n\r\n[Space] - Change Direction\t[X] - Main Menu\t\t[B] - Previous Menu\x1b[0m\r\n\n", RPM);
					clockwiseMotion();
					HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY); // Transmit menu over UART
				}
			}
			else if(rx_data == ' ' && controlMode == 3){

				move_abs32(1,position);
			}
			else if(rx_data == ' ' && controlMode == 4){

				// ReadMotorPosition32(1);
				long targetPos = 65536 * RPM;
				move_rel32(1,targetPos);

			}

		}
		//use the on board user input to exit OR configure uart interrupts - TODO

	}
	HAL_NVIC_DisableIRQ(USART2_IRQn);
	return code;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////CONTROL FUNCTIONS (CAN CHANGE)////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/* Function Name: clockwiseMotion
 * Author: Aidan Drescher
 * Date: 2024-02-21
 * Purpose: Send a command packet to DMM drive to initiate clockwise motion of the shaft with user
 * defined speed
 * */
void clockwiseMotion(){
	if(motorPower == 0){
		return;
	}
	else if(motorPower == 1){
		sendSpeed = speed;
		Turn_const_speed(1,sendSpeed);
		motorDirection = 1;
	}
}

/* Function Name: counterClockwiseMotion
 * Author: Aidan Drescher
 * Date: 2024-02-21
 * Purpose: Send a command packet to DMM drive to initiate counter-clockwise motion of the shaft
 * with user defined speed
 * */
void counterClockwiseMotion(){
	if(motorPower == 0){
		return;
	}
	else if(motorPower == 1){
		sendSpeed = speed*-1;
		Turn_const_speed(1,sendSpeed);
		motorDirection = 2;
	}
}

/* Function Name: speedControl
 * Author: Aidan Drescher
 * Date: 2024-02-21
 * Purpose: Parse incoming speed commands to separate the command type and the real value. Change
 * the multiplier to adjust the scale of the speed
 * */
void speedControl(char* command){
  int index = 0;
  int i = 0;
  char speedVal[3] = {};
  for(i=0; command[i]!='\0'; i++){
	  if(isdigit(command[i])){
	      speedVal[index] = command[i];
	      index++;
	  }
  }
  speed = atoi(speedVal);
  speed = speed * 10; //change multiplier here to determine max speed (motor rated 3000RPM - set to 30)

  if(motorDirection == 1){
	  clockwiseMotion();
  }
  else if(motorDirection == 2){
	  counterClockwiseMotion();
  }
}

/* Function Name: absoluteMove
 * Author: Aidan Drescher
 * Date: 2024-02-21
 * Purpose: Parse incoming absolute move commands to separate the command type and the real value.
 * The function computes the move time based on # of revolutions and calls appropriate functions
 * to execute the move (move_rel32)
 * */
void absoluteMove(char* command){
	int index = 0;
	int i = 0;
	char revCount[3] = {};
	long int noRev = 0;
	int moveTime = 0;
	if(motorPower == 0){
		return;
	}
	else if(motorPower == 1){
		motorDirection = 0;
		Turn_const_speed(1,0);
		HAL_Delay(2000);
		//while timer is not done, wait (still run read data for position display)
		for(i=0; command[i]!='\0'; i++){
			if(isdigit(command[i])){
			    revCount[index] = command[i];
			    index++;
			}
		}
		noRev = atoi(revCount);
		moveTime = 550 * noRev;
		if(command[1] == '-'){
		    noRev = noRev * -65536;
			move_rel32(1,noRev);
			HAL_Delay(moveTime);
		}
		else{
			noRev = noRev * 65536;
			move_rel32(1,noRev);
			HAL_Delay(moveTime);
		}
	}
}

/* Function Name: setMotorSpeed
 * Author: Shayan
 * Date: 2026-02-02
 * Purpose: Helper function to set motor speed for a specific motor ID. Handles user input validation
 * and sends the speed command to the specified motor
 * */
void setMotorSpeed(char motorID){
	char rx_data;
	int rx_index = 0;
	char rx_buffer[30];
	char sendMsg[100];
	char isTyping = 1;
	int speedTemp = 0;

	sprintf(sendMsg, "\r\nEnter Speed (RPM) for Motor %d: ", motorID);
	HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY);

	while(isTyping == 1){

		if (HAL_UART_Receive(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY) == HAL_OK){

			if(rx_data == '\r'){

				rx_buffer[rx_index] = '\0';
				rx_index = 0;
				speedTemp = atoi(rx_buffer);
				//insert software limits on speed here
				if(speedTemp < 0 || speedTemp > 3000){
					HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m", strlen("\r\n\r\n\x1b[1;31mInvalid input!\r\n\r\nSpeed must be between 0 and 3000 RPM!\n\n\r\x1b[0m"), HAL_MAX_DELAY);
					memset(rx_buffer,' ',30);
					HAL_Delay(250);
					sprintf(sendMsg, "\r\nEnter Speed (RPM) for Motor %d: ", motorID);
					HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY);
				}
				else{
					isTyping = 0;
					speed = speedTemp;
					sprintf(sendMsg,"\r\n\n\x1b[1mMotor %d speed set to %d RPM\x1b[0m\r\n", motorID, speedTemp);
					HAL_UART_Transmit(&huart2, (uint8_t *)sendMsg, strlen(sendMsg), HAL_MAX_DELAY);
					Turn_const_speed(motorID, speed);
				}

			}
			else{

				rx_buffer[rx_index++] = rx_data;

                if (rx_index >= 30){
                    rx_index = 0;
                }

                HAL_UART_Transmit(&huart2, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY);

			}

		}

	}
}

/* Function Name: absMoveSetup
 * Author: Aidan Drescher
 * Date: 2024-04-10
 * Purpose: Ensures positional accuracy before relative moves
 * */
void absMoveSetup(){

	for(int i=0; i<45; i++){
		//sendString();
		HAL_Delay(75);
	}
}

/* Function Name: goHome
 * Author: Aidan Drescher
 * Date: 2024-03-19
 * Purpose: Sends the motor back to 0 pulses on the encoder
 * */
int goHome(){

  if(motorPower == 0){
    return 5;
  }
  else if(motorPower == 1){

    motorDirection = 0;
    Turn_const_speed(1,0);
    ReadMotorPosition32(1);

    if(Motor_Pos32 > 400000){
      while(Motor_Pos32 > 400000){
        //sendString();
    	ReadMotorPosition32(1);
        Turn_const_speed(1,-750);
      }
      while(Motor_Pos32 > 200000 && Motor_Pos32 <= 400000){
        //sendString();
    	ReadMotorPosition32(1);
        Turn_const_speed(1,-250);
      }
      Turn_const_speed(1,0);
      absMoveSetup();
      ReadMotorPosition32(1);
      move_rel32(1,-Motor_Pos32);
      return 1;
    }
    else if(Motor_Pos32 < -400000){
      while(Motor_Pos32 < -400000){
        //sendString();
    	ReadMotorPosition32(1);
        Turn_const_speed(1,750);
      }
      while(Motor_Pos32 < -200000 && Motor_Pos32 >= -400000){
        //sendString();
    	ReadMotorPosition32(1);
        Turn_const_speed(1,250);
      }
      Turn_const_speed(1,0);
      absMoveSetup();
      ReadMotorPosition32(1);
      move_rel32(1,-Motor_Pos32);
      return 1;
    }
    else{
      Turn_const_speed(1,0);
      absMoveSetup();
      ReadMotorPosition32(1);
      move_rel32(1,-Motor_Pos32);
      return 1;
    }


  }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DATA READING & DISPLAY FUNCTIONS/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

/* Function Name: sendString
 * Author: Aidan Drescher
 * Date: 2024-03-19
 * Purpose: Sends motor position and speed to the screen
 * */
void sendString(){

  ReadMotorPosition32(1);
  classifyReadString(Motor_Pos32,'p',outputString);
  HAL_Delay(25);
  HAL_UART_Transmit(&huart2, (uint8_t*)outputString, strlen(outputString), HAL_MAX_DELAY);

  ReadMotorSpeed32(1);
  classifyReadString(Motor_Speed32,'s',outputString);
  HAL_Delay(25);
  HAL_UART_Transmit(&huart2, (uint8_t*)outputString, strlen(outputString), HAL_MAX_DELAY);

}

/* Function Name: classifyReadString
 * Author: Aidan Drescher
 * Date: 2024-04-10
 * Purpose: Decodes data strings sent from drive
 * */
void classifyReadString(long value, char charToAdd, char* result){

  char buffer[25];

  ltoa(value,buffer,10);

  //Serial.println("CLASSIFYING STRING");

  strcpy(result, buffer);

  int len = strlen(result);

  result[len] = charToAdd;
  result[len + 1] = '\0'; // Null-terminate the string

  //Serial.println("STRING CLASSIFIED");

}

/* Function Name: ltoa
 * Author: Aidan Drescher
 * Date: 2024-04-11
 * Purpose: Acts as the stdlib.h ltoa function as STM32CubeIDE does not support the native
 * Converts long interger value to string for proper data type classification.
 * */
char* ltoa(long value, char* result, int base) {

	int i = 0;
	int isNegative = 0;
	// Check for invalid base
    if (base < 2 || base > 36) {
        *result = '\0';
        return result;
    }
    // Handle negative numbers

    if (value < 0) {
        isNegative = 1;
        value = -value;
    }
    // Convert the number to string in reverse order
    do {
        int digit = value % base;
        result[i++] = (digit < 10) ? digit + '0' : digit + 'A' - 10;
        value /= base;
    } while (value > 0);

    // Add negative sign if necessary
    if (isNegative) {
        result[i++] = '-';
    }
    // Terminate the string
    result[i] = '\0';
    // Reverse the string
    int len = i;
    for (int j = 0; j < len / 2; j++) {
        char temp = result[j];
        result[j] = result[len - j - 1];
        result[len - j - 1] = temp;
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TIMER DEPENDANT FUNCTIONS////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

/* Function Name: timeDelay
 * Author: Aidan Drescher
 * Date: 2024-02-21
 * Purpose: Uses Timer 3 to generate a delay so interrupts remain free
 * */
void timeDelay(int delay){

	__HAL_TIM_SET_COUNTER(&htim3,0);
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, delay);
	HAL_TIM_OC_Start_IT(&htim3, TIM_CHANNEL_1);
	while(!(delayDone));
	delayDone = 0;
	HAL_TIM_OC_Stop_IT(&htim3, TIM_CHANNEL_1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////DYN DRIVE FUNCTIONS (DONT CHANGE)////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

/* Function Name: Drive_Reset_RS232
 * Author: Aidan Drescher
 * Date: 2024-05-17
 * Purpose: Prepares a relative move packet to send to DMM drive
 * */
// void Drive_Reset_RS232(){
// 	timeDelay(500);
// 	Global_Func = (char)General_Read;
// 	Send_Package(1,Is_DriveReset);
// }

// void Drive_Enable_RS232(){
// 	timeDelay(500);
// 	Global_Func = (char)General_Read;
// 	Send_Package(1,Is_DriveEnable);
// }

// void Drive_Disable_RS232(){
// 	timeDelay(500);
// 	Global_Func = (char)General_Read;
// 	Send_Package(1,Is_DriveDisable);
// }
/* Function Name: move_rel32
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares a relative move packet to send to DMM drive
 * */
void move_rel32(char ID, long pos)
{
  char Axis_Num = ID;
  Global_Func = (char)Go_Relative_Pos;
  Send_Package(Axis_Num, pos);
}

/* Function Name: ReadMotorTorqueCurrent
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Executes a read from the DMM drive to obtain motor torque
 * */
void ReadMotorTorqueCurrent(char ID)
{
  Global_Func = General_Read;
  Send_Package(ID , Is_TrqCurrent);
  MotorTorqueCurrentReady_Flag = 0xff;
  while(MotorTorqueCurrentReady_Flag != 0x00)
  {
    ReadPackage();
  }
}

/* Function Name: ReadMotorPosition32
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Executes a read from the DMM drive to obtain motor position in encoder pulses
 * */
void ReadMotorPosition32(char ID)
{
  Global_Func = (char)General_Read;
  Send_Package(ID , Is_AbsPos32);
  MotorPosition32Ready_Flag = 0xff;
  while(MotorPosition32Ready_Flag != 0x00)
  {
    ReadPackage();
  }
}

/* Function Name: ReadMotorSpeed32
 * Author: Aidan Drescher
 * Date: 2024-04-10
 * Purpose: Executes a read from the DMM drive to obtain motor speed in RPM
 * */
void ReadMotorSpeed32(char ID)
{
  Global_Func = (char)General_Read;
  Send_Package(ID, Read_HighSpeed);
  MotorSpeed32Ready_Flag = 0xff;
//   while(MotorSpeed32Ready_Flag != 0x00)
//   {
    // ReadPackage();
//   }
}
/* Function Name: ReadMotorTorqueCurrent
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares an absolute move packet to send to DMM drive
 * */
void move_abs32(char MotorID, long Pos32)
{
  char Axis_Num = MotorID;
  Global_Func = (char)Go_Absolute_Pos;
  Send_Package(Axis_Num, Pos32);
}

/* Function Name: Turn_const_speed
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares and sends a data packet to execute a constant speed command
 * */
void Turn_const_speed(char ID, long spd)
{
      char Axis_Num = ID;
      Global_Func = 0x0a;
      Send_Package(Axis_Num, spd);
}


/* Function Name: classifyReadString
 * Author: Aidan Drescher
 * Date: 2024-04-11
 * Purpose: Reads data from drive.
 * */
void ReadPackage(void) {

  signed int c, cif;

  while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE)) {
    HAL_UART_Receive(&huart1, &c, 1, HAL_MAX_DELAY); // Receive one byte from UART
    InputBuffer[InBfTopPointer] = c; // Load InputBuffer with received packets
    InBfTopPointer++;
  }

  while (InBfBtmPointer != InBfTopPointer) {
    c = InputBuffer[InBfBtmPointer];
    InBfBtmPointer++;
    cif = c & 0x80;
    if (cif == 0) {
      Read_Num = 0;
      Read_Package_Length = 0;
    }
    if (cif == 0 || Read_Num > 0) {
      Read_Package_Buffer[Read_Num] = c;
      Read_Num++;
      if (Read_Num == 2) {
        cif = c >> 5;
        cif = cif & 0x03;
        Read_Package_Length = 4 + cif;
        c = 0;
      }
      if (Read_Num == Read_Package_Length) {
        Get_Function(); // Assuming Get_Function() is defined elsewhere
        Read_Num = 0;
        Read_Package_Length = 0;
      }
    }
  }
}

/* Function Name: Get_Function
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Decodes the desired function code and toggles related flag. Initiates data processing
 * */
void Get_Function(void) {
  char ID, ReceivedFunction_Code, CRC_Check;
  long Temp32;
  ID = Read_Package_Buffer[0] & 0x7f;
  ReceivedFunction_Code = Read_Package_Buffer[1] & 0x1f;
  CRC_Check = 0;

  for (int i = 0; i < Read_Package_Length - 1; i++) {
    CRC_Check += Read_Package_Buffer[i];
  }

  CRC_Check ^= Read_Package_Buffer[Read_Package_Length - 1];
  CRC_Check &= 0x7f;
  if (CRC_Check != 0) {
  }
  else {
    switch (ReceivedFunction_Code)
    {
      case  Is_AbsPos32:
        Motor_Pos32 = Cal_SignValue(Read_Package_Buffer);
        MotorPosition32Ready_Flag = 0x00;
        break;
    //   case  Is_MotorSpeed:
    //     Motor_Speed32 = Cal_SignValue(Read_Package_Buffer);
    //     MotorSpeed32Ready_Flag = 0x00;
    //     break;
      case  Is_TrqCurrent:
	      MotorTorqueCurrent = Cal_SignValue(Read_Package_Buffer);
	    break;
      case  Is_Status:
        Driver_Status = (char)Cal_SignValue(Read_Package_Buffer);
        // Driver_Status=drive status byte data
        break;
      case  Is_Config:
        Temp32 = Cal_Value(Read_Package_Buffer);
        //Driver_Config = drive configuration setting
        break;
      case  Is_MainGain:
        Driver_MainGain = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_MainGain = Driver_MainGain & 0x7f;
        break;
      case  Is_SpeedGain:
        Driver_SpeedGain = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_SpeedGain = Driver_SpeedGain & 0x7f;
        break;
      case  Is_IntGain:
        Driver_IntGain = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_IntGain = Driver_IntGain & 0x7f;
        break;
      case  Is_TrqCons:
        Driver_TrqCons = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_TrqCons = Driver_TrqCons & 0x7f;
        break;
      case  Is_HighSpeed:
        Driver_HighSpeed = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_HighSpeed = Driver_HighSpeed & 0x7f;
        break;
      case  Is_HighAccel:
        Driver_HighAccel = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_HighAccel = Driver_HighAccel & 0x7f;
        break;
      case  Is_Driver_ID:
        Driver_ReadID = ID;
        break;
      case  Is_Pos_OnRange:
        Driver_OnRange = (char)Cal_SignValue(Read_Package_Buffer);
        Driver_OnRange = Driver_OnRange & 0x7f;
        break;
    }
  }
}
/* Function Name: Cal_SignValue
 * Author: Aidan Drescher
 * Date: 2024-04-17
 * Purpose: Interprets the correct mathematical sign (pos/neg) from the 2^18 value
 * (splits into -2^17 ~ 2^17 - 1)
 * */
int32_t Cal_SignValue(unsigned char* One_Package) {
    char Package_Length, i;
    int32_t Lcmd;

    // Determine package length based on the second byte
    Package_Length = 4 + ((One_Package[1] >> 5) & 0x03);

    // Initialize the command value
    Lcmd = (One_Package[2] & 0x7F); // Mask out the sign bit

    // Extract the sign bit from the first byte
    int sign_bit = (One_Package[2] & 0x40) ? 1 : 0;

    // Sign extension if necessary
    if (sign_bit) {
        Lcmd |= 0xFFFFFF80; // Sign extend to 32 bits
    }

    // Process the remaining bytes
    for (i = 3; i < Package_Length - 1; i++) {
        Lcmd = (Lcmd << 7) | (One_Package[i] & 0x7F);
    }

    return Lcmd; // Lcmd: -2^17 ~ 2^17 - 1
}

/* Function Name: Cal_Value
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose:
 * */
long Cal_Value(unsigned char* One_Package)
{
  char Package_Length,OneChar,i;
  long Lcmd;
  OneChar = One_Package[1];
  OneChar = OneChar>>5;
  OneChar = OneChar&0x03;
  Package_Length = 4 + OneChar;

  OneChar = One_Package[2];   /*First byte 0x7f, bit 6 reprents sign      */
  OneChar &= 0x7f;
  Lcmd = (long)OneChar;     /*Sign extended to 32bits           */
  for(i=3;i<Package_Length-1;i++)
  {
    OneChar = One_Package[i];
    OneChar &= 0x7f;
    Lcmd = Lcmd<<7;
    Lcmd += OneChar;
  }
  return(Lcmd);         /*Lcmd : -2^27 ~ 2^27 - 1           */
}

/* Function Name: Send_Package
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Prepares and sends a data packet
 * */
void Send_Package(char ID , long Displacement) {
  unsigned char B[8], Package_Length, Function_Code;
  long TempLong;
  B[1] = B[2] = B[3] = B[4] = B[5] = (unsigned char)0x80;
  B[0] = ID & 0x7f;
  Function_Code = Global_Func & 0x1f;
  TempLong = Displacement & 0x0fffffff; //Max 28bits
  B[5] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[4] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[3] += (unsigned char)TempLong & 0x0000007f;
  TempLong = TempLong >> 7;
  B[2] += (unsigned char)TempLong & 0x0000007f;
  Package_Length = 7;
  TempLong = Displacement;
  TempLong = TempLong >> 20;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //Three byte data
    B[2] = B[3];
    B[3] = B[4];
    B[4] = B[5];
    Package_Length = 6;
  }
  TempLong = Displacement;
  TempLong = TempLong >> 13;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //Two byte data
    B[2] = B[3];
    B[3] = B[4];
    Package_Length = 5;
  }
  TempLong = Displacement;
  TempLong = TempLong >> 6;
  if (( TempLong == 0x00000000) || ( TempLong == 0xffffffff)) { //One byte data
    B[2] = B[3];
    Package_Length = 4;
  }
  B[1] += (Package_Length - 4) * 32 + Function_Code;
  Make_CRC_Send(Package_Length, B);
}

/* Function Name: Make_CRC_Send
 * Author: Tianyu Li
 * Date: 2019-02-21
 * Purpose: Checks packet and transmits
 * */
void Make_CRC_Send(unsigned char Plength, unsigned char* B) {
  unsigned char Error_Check = 0;
  char RS232_HardwareShiftRegister;

  for (int i = 0; i < Plength - 1; i++) {
    OutputBuffer[OutBfTopPointer] = B[i];
    OutBfTopPointer++;
    Error_Check += B[i];
  }
  Error_Check = Error_Check | 0x80;
  OutputBuffer[OutBfTopPointer] = Error_Check;
  OutBfTopPointer++;
  while (OutBfBtmPointer != OutBfTopPointer) {
    RS232_HardwareShiftRegister = OutputBuffer[OutBfBtmPointer];
    //Serial.print("RS232_HardwareShiftRegister: ");
    //Serial.println(RS232_HardwareShiftRegister, DEC);
    //use &huart1 for production
    HAL_UART_Transmit(&huart1, &RS232_HardwareShiftRegister, 1, HAL_MAX_DELAY);
    OutBfBtmPointer++; // Change to next byte in OutputBuffer to send
  }
}

/* USER CODE END 4 */

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM14 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM14) {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
