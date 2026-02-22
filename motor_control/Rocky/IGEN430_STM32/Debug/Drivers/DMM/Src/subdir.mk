################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/DMM/Src/DMM_driver.c 

OBJS += \
./Drivers/DMM/Src/DMM_driver.o 

C_DEPS += \
./Drivers/DMM/Src/DMM_driver.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/DMM/Src/%.o Drivers/DMM/Src/%.su Drivers/DMM/Src/%.cyclo: ../Drivers/DMM/Src/%.c Drivers/DMM/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m3 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L152xE -c -I../Core/Inc -I../Drivers/STM32L1xx_HAL_Driver/Inc -I../Drivers/STM32L1xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L1xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfloat-abi=soft -mthumb -o "$@"

clean: clean-Drivers-2f-DMM-2f-Src

clean-Drivers-2f-DMM-2f-Src:
	-$(RM) ./Drivers/DMM/Src/DMM_driver.cyclo ./Drivers/DMM/Src/DMM_driver.d ./Drivers/DMM/Src/DMM_driver.o ./Drivers/DMM/Src/DMM_driver.su

.PHONY: clean-Drivers-2f-DMM-2f-Src

