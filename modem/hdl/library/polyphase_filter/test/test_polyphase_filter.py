# Simple tests for an adder module
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave
import random
import sys

import dut_control_polyphase_filter

sys.path.insert(0, '../../../../python/cocotb')
import helper_functions

sys.path.insert(0, '../../../../python/library')
import comms_filters




@cocotb.test()
def rrc(dut):
	"""
		Test a Root Raised Cosine filter.

		Load the filter coefficients with a RRC filter kernel then
		send a unit pulse to find the impulse response of the filter
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	yield dut_control_obj.init()

	# create the coefficients
	coeffs = comms_filters.rrcosfilter(	N = dut_control_obj.NUMBER_TAPS, 
										alpha  = 0.5, 
										Ts = 1, 
										Fs = dut_control_obj.RATE_CHANGE)[1]
	# scale the coefficients 
	coeff_max = sum(coeffs)
	coeffs = [int(coeff * ((2**(dut_control_obj.COEFFS_WIDTH-1)-1)/coeff_max)) for coeff in coeffs]

	# write in the coefficients
	yield dut_control_obj.coefficients_write(coeffs)

	# send in a first pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")






@cocotb.test()
def basic(dut):
	"""
		Test basic functionality
		
		Use a filter with incrementing integer coefficients and then
		find the impulse response to ensure it matches.
	"""

	# setup the DUT
	dut_control_obj = dut_control_polyphase_filter.dut_control_polyphase_filter(dut)
	yield dut_control_obj.init()

	# create some simple coefficients
	coeffs = [_ for _ in range(int(-dut_control_obj.NUMBER_TAPS/2),int(dut_control_obj.NUMBER_TAPS/2))]

	# write in the coefficients
	yield dut_control_obj.coefficients_write(coeffs)
	dut._log.info("Coefficients have been written to the DUT.")

	# send in a first pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)
	dut._log.info("First impulse has been sent.")

	# send in a second pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)
	dut._log.info("Second impulse has been sent.")

	# send in a third pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)
	dut._log.info("Third impulse has been sent.")

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")