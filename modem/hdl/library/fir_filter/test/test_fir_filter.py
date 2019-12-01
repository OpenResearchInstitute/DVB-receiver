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
import matplotlib.pyplot as plt

import dut_control_fir_filter

sys.path.insert(0, '../../../../python/cocotb')
import helper_functions




@cocotb.test()
def basic(dut):
	"""Test basic functionality"""

	# setup the DUT
	dut_control_obj = dut_control_fir_filter.dut_control_fir_filter(dut)
	yield dut_control_obj.init()

	# set the system parameters
	coeffs = [_ for _ in range(int(-dut_control_obj.NUMBER_TAPS/2),int(dut_control_obj.NUMBER_TAPS/2))]

	# convert negative numbers to twos complement and arrange for the polyphase structure
	coeffs_fixedpoint = helper_functions.signed_to_fixedpoint(coeffs, dut_control_obj.COEFFS_WIDTH)

	# write in the coefficients
	yield dut_control_obj.coefficients_write(coeffs_fixedpoint)

	# send in a first pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)
	yield dut_control_obj.wait(16)

	# send in a second pulse
	yield dut_control_obj.data_out_read_enable()
	yield dut_control_obj.impulse_response(coeffs)
	yield dut_control_obj.wait(16)

	# if we have reached this point the test is a success
	dut._log.info("Test has completed with no failures detected")