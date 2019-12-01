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
import numpy as np

import dut_control_apsk_modulator

sys.path.insert(0, '../../../../python/cocotb')
import helper_functions







@cocotb.test()
def basic(dut):
	"""
		Test basic functionality
	"""

	# setup the DUT
	dut_control_obj = dut_control_apsk_modulator.dut_control_apsk_modulator(dut)
	yield dut_control_obj.init()

	# dut_control_obj.PLOT = False
	dut_control_obj.PLOT = True

	# select the modulation type
	# yield dut_control_obj.set_modulation(["216", "QPSK 11/45"])
	# yield dut_control_obj.set_modulation(["150", "16APSK 8/15-L"])
	# yield dut_control_obj.set_modulation(["214", "256APSK 3/4"])
	yield dut_control_obj.set_modulation(["142", "8PSK 23/36"])

	# repeat multiple test
	for i in range(dut_control_obj.NUMBER_REPEAT_TESTS):

		dut_control_obj.dut._log.info("Performing test number %d" % (i+1))

		# ready the output port to receive the data
		yield dut_control_obj.data_out_read_enable()

		# create some random data and write to the DUT
		data_length = 8
		data_in = [np.random.randint(2**dut_control_obj.INPUT_WIDTH) for _ in range(data_length)]
		yield dut_control_obj.data_in_write(data_in)

		# parse the output data
		yield dut_control_obj.axiss_read_handle.join()
		dut_control_obj.data_out_read_parse()

		# test the received data against the python implementation
		dut_control_obj.test_modulation(data_in)

		# wait for a small amount of time
		yield dut_control_obj.wait(8)




@cocotb.test()
def all_modulations(dut):
	"""
		Loop through all modulation types, testing them
	"""

	# setup the DUT
	dut_control_obj = dut_control_apsk_modulator.dut_control_apsk_modulator(dut)
	yield dut_control_obj.init()

	dut_control_obj.PLOT = False
	# dut_control_obj.PLOT = True


	modulation_code_list = list(dut_control_obj.modulation_definition.keys())
	modulation_code_list.remove("modulation_type_depth")

	# loop through all modualtion types
	for modulation_code in modulation_code_list:

		print(modulation_code)

		# pull out the names of the modulation
		modulation = dut_control_obj.modulation_definition[modulation_code]
		
		print(modulation)

		modulation_name = list(modulation.keys())[0]

		print(modulation_name)

		# select the modulation type
		dut_control_obj.dut._log.info("Setting up modulation %s : %s" % (modulation_code, modulation_name))
		yield dut_control_obj.set_modulation([modulation_code, modulation_name])

		# repeat multiple test
		for i in range(dut_control_obj.NUMBER_REPEAT_TESTS):

			dut_control_obj.dut._log.info("Performing test number %d" % (i+1))

			# ready the output port to receive the data
			yield dut_control_obj.data_out_read_enable()

			# create some random data and write to the DUT
			data_length = 8
			data_in = [np.random.randint(2**dut_control_obj.INPUT_WIDTH) for _ in range(data_length)]
			yield dut_control_obj.data_in_write(data_in)

			# parse the output data
			yield dut_control_obj.axiss_read_handle.join()
			dut_control_obj.data_out_read_parse()

			# test the received data against the python implementation
			dut_control_obj.test_modulation(data_in)
			
			# wait for a small amount of time
			yield dut_control_obj.wait(8)
