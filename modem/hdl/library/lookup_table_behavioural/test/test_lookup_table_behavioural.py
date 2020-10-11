import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave
import random
import sys

import dut_control_lookup_table



@cocotb.test()
def write_read_counter(dut):
	"""
		Fill the memory with a counter and read it back to ensure
		it matches
	"""

	# setup the DUT
	dut_control_obj = dut_control_lookup_table.dut_control_lookup_table(dut)
	yield dut_control_obj.init()

	# create a counter input data
	input_data = [_ for _ in range(dut_control_obj.MEMORY_DEPTH)]

	# write in the input data
	yield dut_control_obj.memory_write(input_data)

	# send in a first pulse
	output_data = yield dut_control_obj.memory_read()

	# test that the input and output data match
	if input_data != output_data:
		print("Input: ", input_data)
		print("Output: ", output_data)
		raise TestFailure("Data read from memory does not match that written to memory!")
	else:
		dut_control_obj.dut._log.info("Data written and read from memory correctly")
