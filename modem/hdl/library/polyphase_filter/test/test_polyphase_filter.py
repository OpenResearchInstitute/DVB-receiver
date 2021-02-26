# Simple tests for an adder module
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave
from cocotbext.axi import AxiLiteMaster
import numpy as np
import matplotlib.pyplot as plt
import random
import sys


import dut_control_polyphase_filter

sys.path.insert(0, '../../../../python/cocotb')
import helper_functions

sys.path.insert(0, '../../../../python/library')
import comms_filters




@cocotb.test()
async def rrc(dut):
	"""
		Test a Root Raised Cosine filter.

		Load the filter coefficients with a RRC filter kernel then
		send a unit pulse to find the impulse response of the filter
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	await dut_control_obj.init()

	# create the coefficients
	coeffs = comms_filters.rrcosfilter(	N = dut_control_obj.NUMBER_TAPS, 
										alpha  = 0.5, 
										Ts = 1, 
										Fs = dut_control_obj.RATE_CHANGE)[1]
	# scale the coefficients 
	coeff_max = sum(coeffs)
	coeffs = [int(coeff * ((2**(dut_control_obj.COEFFS_WIDTH-1)-1)/coeff_max)) for coeff in coeffs]

	# write in the coefficients
	await dut_control_obj.coefficients_write(coeffs)

	# send in a first pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.impulse_response(coeffs)

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")






@cocotb.test()
async def basic(dut):
	"""
		Test basic functionality
		
		Use a filter with incrementing integer coefficients and then
		find the impulse response to ensure it matches.
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	await dut_control_obj.init()

	# create some simple coefficients
	coeffs = [_ for _ in range(int(-dut_control_obj.NUMBER_TAPS/2),int(dut_control_obj.NUMBER_TAPS/2))]

	# write in the coefficients
	await dut_control_obj.coefficients_write(coeffs)
	dut._log.info("Coefficients have been written to the DUT.")

	# send in a first pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.impulse_response(coeffs)
	dut._log.info("First impulse has been sent.")

	# send in a second pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.impulse_response(coeffs)
	dut._log.info("Second impulse has been sent.")

	# send in a third pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.impulse_response(coeffs)
	dut._log.info("Third impulse has been sent.")

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")





@cocotb.test()
async def gnuradio_input(dut):
	"""
		Test a Root Raised Cosine filter.

		Load the filter coefficients with a RRC filter kernel then
		send a unit pulse to find the impulse response of the filter
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	await dut_control_obj.init()

	# create the coefficients
	# coeffs = comms_filters.rrcosfilter(	N = dut_control_obj.NUMBER_TAPS, 
	# 									alpha  = 0.5, 
	# 									Ts = 1, 
	# 									Fs = dut_control_obj.RATE_CHANGE)[1]

	# create the coefficients
	coeffs = comms_filters.rrcosfilter(	N = 32,
										alpha  = 0.5, 
										# Ts = 1, 
										Ts = 2, 
										# Fs = 1)[1]
										Fs = 2)[1]

	# scale the coefficients 
	coeff_max = sum(coeffs)
	coeffs = [int(coeff * ((2**(dut_control_obj.COEFFS_WIDTH-1)-1)/coeff_max)) for coeff in coeffs]
	
	for coeff in coeffs:

		if coeff < 0:
			print("%04X" % (coeff + 2**16))
		else:
			print("%04X" % coeff)

	# write in the coefficients
	await dut_control_obj.coefficients_write(coeffs)
	
	# read the GNURadio data
	gnuradio_data = np.fromfile(open("/home/tom/repositories/dvb_fpga/gnuradio_data/FECFRAME_NORMAL_MOD_QPSK_C1_2/plframe_pilots_on_fixed_point.bin"), dtype=np.int16)
	gnuradio_data = gnuradio_data[:256]

	input_data = []
	for i in range(0,int(len(gnuradio_data)/2),2):
		input_data.append( int(gnuradio_data[i]/1) )


	# send in a first pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.axism_data_in.write(input_data)

	# read the output
	await dut_control_obj.axiss_read_handle.join()

	received_data = helper_functions.fixedpoint_to_signed(dut_control_obj.axiss_data_out.data, dut_control_obj.COEFFS_WIDTH)
	# plt.plot([_*8 for _ in range(len(input_data)+2)], [0]*2 + input_data)
	plt.plot([_*2 for _ in range(len(input_data)+8)], [0]*8 + input_data)
	plt.plot(received_data)
	plt.show()
	# received_data = helper_functions.fixedpoint_to_signed(self.axiss_data_out.data, self.COEFFS_WIDTH)

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")




@cocotb.test()
async def gnuradio_impulse(dut):
	"""
		Test a Root Raised Cosine filter.

		Load the filter coefficients with a RRC filter kernel then
		send a unit pulse to find the impulse response of the filter
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	await dut_control_obj.init()

	# # create the coefficients
	# coeffs = comms_filters.rrcosfilter(	N = dut_control_obj.NUMBER_TAPS, 
	# 									alpha  = 0.5, 
	# 									Ts = 1, 
	# 									Fs = dut_control_obj.RATE_CHANGE)[1]

	# create the coefficients
	coeffs = comms_filters.rrcosfilter(	N = 101, 
										alpha  = 0.2, 
										Ts = 1, 
										Fs = 2)[1]

	print('len(coeffs)', len(coeffs))


	# scale the coefficients 
	coeff_max = sum(coeffs)
	coeffs = [int(coeff *((2**(dut_control_obj.COEFFS_WIDTH-1)-1)/coeff_max)) for coeff in coeffs]
	
	for coeff in coeffs:

		if coeff < 0:
			print("%04X" % (coeff + 2**16))
		else:
			print("%04X" % coeff)





	# write in the coefficients
	await dut_control_obj.coefficients_write(coeffs)
	
	# read the GNURadio data
	# gnuradio_data = np.fromfile(open("gnuradio_filterout.bin"), dtype=np.complex64)
	gnuradio_data = np.fromfile(open("gnuradio_filterout_short.bin"), dtype=np.int16)
	gnuradio_data = gnuradio_data[256:256+404]

	gnr_impulse = []
	for i in range(0,int(len(gnuradio_data)/2),2):
		gnr_impulse.append( int(gnuradio_data[i]/1) + 1j*int(gnuradio_data[i+1]/1) )

	print(gnr_impulse)

	# plt.plot(gnr_impulse)
	# plt.show()


	# send in a first pulse
	await dut_control_obj.data_out_read_enable()
	await dut_control_obj.axism_data_in.write([int(2**(dut_control_obj.DATA_WIDTH-2))])

	# read the output
	await dut_control_obj.axiss_read_handle.join()

	received_data = helper_functions.fixedpoint_to_signed(dut_control_obj.axiss_data_out.data, dut_control_obj.COEFFS_WIDTH)
	plt.plot(gnr_impulse)
	plt.plot(coeffs)
	plt.plot(received_data)
	plt.show()
	# received_data = helper_functions.fixedpoint_to_signed(self.axiss_data_out.data, self.COEFFS_WIDTH)

	# # if we have reached this point the test is a success
	# dut._log.info("Test has completed with no failures detected")