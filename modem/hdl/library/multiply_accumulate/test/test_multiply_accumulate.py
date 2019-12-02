# Simple tests for an adder module
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
import random

clock_PERIOD = 10


@cocotb.test()
def basic(dut):
	"""Test basic functionality"""

	# setup clocking
	clock_handle = cocotb.fork(Clock(dut.clock, 100).start())
	clock_rising = yield RisingEdge(dut.clock)


	# set up the MAC
	dut.data_in = 0
	dut.coefficient_in = 0
	dut.carry_in = 0
	dut.ce_calculate = 1
	dut.ce_coefficient = 1
	dut.carry_in = 0
	dut.data_carry = 0

	dut.op_mode = int(0b0000101)
	dut.in_mode = int(0b00001)



	# reset the device
	dut.reset = 1
	dut.reset_coefficient = 1
	yield clock_rising
	dut.reset = 0
	dut.reset_coefficient = 0
	dut.DSP48E1_inst.GSR = int(0b0)
	yield clock_rising

	yield clock_rising
	yield clock_rising
	yield clock_rising
	yield clock_rising
	yield clock_rising

	# set the inputs
	dut.data_in = 4*2**10
	dut.coefficient_in = 4*2**10
	yield clock_rising
	yield clock_rising
	dut.carry_in = 4
	yield clock_rising
	yield clock_rising
	yield clock_rising


	# remove the coefficient input 
	dut.ce_coefficient = 0
	dut.coefficient_in = 0
	yield clock_rising


	# delay
	for i in range(8):
		yield clock_rising


	# print the output
	dut._log.info("data_out = %d" % dut.data_out.value.integer)
	# dut._log.info("data_in_carry = %d" % dut.data_in_carry.value.integer)