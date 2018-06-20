################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../bch_decode.cu \
../de_map_int.cu \
../deinterleaver.cu \
../descrambler.cu \
../dvbs2_rx_control.cu \
../ldpc_decoder2.cu \
../pl_vl_snr.cu \
../preamble.cu \
../receiver_a.cu \
../receiver_b.cu \
../testbench.cu 

CPP_SRCS += \
../bb_header_decoder.cpp \
../bch_decode_b.cpp \
../const_tables.cpp \
../data_output.cpp \
../dvb2_ldpc_tables.cpp \
../equaliser.cpp \
../hardware.cpp \
../ldpc_x2_tables.cpp \
../lime.cpp \
../main.cpp \
../noise.cpp \
../pl_decoder.cpp \
../pluto.cpp \
../poly_inv_tab.cpp \
../rrc.cpp \
../stats.cpp \
../zigzag.cpp 

OBJS += \
./bb_header_decoder.o \
./bch_decode.o \
./bch_decode_b.o \
./const_tables.o \
./data_output.o \
./de_map_int.o \
./deinterleaver.o \
./descrambler.o \
./dvb2_ldpc_tables.o \
./dvbs2_rx_control.o \
./equaliser.o \
./hardware.o \
./ldpc_decoder2.o \
./ldpc_x2_tables.o \
./lime.o \
./main.o \
./noise.o \
./pl_decoder.o \
./pl_vl_snr.o \
./pluto.o \
./poly_inv_tab.o \
./preamble.o \
./receiver_a.o \
./receiver_b.o \
./rrc.o \
./stats.o \
./testbench.o \
./zigzag.o 

CU_DEPS += \
./bch_decode.d \
./de_map_int.d \
./deinterleaver.d \
./descrambler.d \
./dvbs2_rx_control.d \
./ldpc_decoder2.d \
./pl_vl_snr.d \
./preamble.d \
./receiver_a.d \
./receiver_b.d \
./testbench.d 

CPP_DEPS += \
./bb_header_decoder.d \
./bch_decode_b.d \
./const_tables.d \
./data_output.d \
./dvb2_ldpc_tables.d \
./equaliser.d \
./hardware.d \
./ldpc_x2_tables.d \
./lime.d \
./main.d \
./noise.d \
./pl_decoder.d \
./pluto.d \
./poly_inv_tab.d \
./rrc.d \
./stats.d \
./zigzag.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -O3 -Xcompiler $(shell pkg-config --cflags gtk+-3.0) -gencode arch=compute_52,code=sm_52  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -O3 -Xcompiler $(shell pkg-config --cflags gtk+-3.0) --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -O3 -Xcompiler $(shell pkg-config --cflags gtk+-3.0) -gencode arch=compute_52,code=sm_52  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -O3 -Xcompiler $(shell pkg-config --cflags gtk+-3.0) --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


