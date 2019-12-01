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
	dut.data_in_A = 0
	dut.data_in_D = 0
	dut.coefficient_in = 0
	dut.carry_in = 0
	dut.ce_calculate = 1
	dut.ce_coefficient = 1
	dut.carry_in = 0

	dut.op_mode = int(0b0110101)
	dut.in_mode = int(0b10101)

	# setup the DSP48
	# dut.A = int(0b000000000000000000000000000000)
	# dut.ACIN = int(0b000000000000000000000000000000)
	# dut.ALUMODE = int(0b0000) 
	# dut.B = int(0b000000000000000000)  
	# dut.BCIN = int(0b000000000000000000) 
	# dut.C = int(0b000000000000000000000000000000000000000000000000)
	# dut.CARRYCASCIN  = int(0b0)
	# dut.CARRYIN  = int(0b0)
	# dut.CARRYINSEL = int(0b000)
	# dut.CEA1 = int(0b1)
	# dut.CEA2 = int(0b1)
	# dut.CEAD = int(0b1)
	# dut.CEALUMODE = int(0b1)
	# dut.CEB1 = int(0b1)
	# dut.CEB2 = int(0b1)
	# dut.CEC = int(0b1)
	# dut.CECARRYIN = int(0b1)
	# dut.CECTRL = int(0b1)
	# dut.CED = int(0b1)
	# dut.CEINMODE = int(0b1)
	# dut.CEM = int(0b1)
	# dut.CEP = int(0b1)
	# dut.D = int(0b0000000000000000000000000)
	# dut.MULTSIGNIN = int(0b0)
	# # dut.OPMODE = int(0b0000101)
	# dut.PCIN = int(0b000000000000000000000000000000000000000000000000)
	# dut.RSTA = int(0b0)
	# dut.RSTALLCARRYIN = int(0b0)
	# dut.RSTALUMODE = int(0b0)
	# dut.RSTB = int(0b0)
	# dut.RSTC = int(0b0)
	# dut.RSTCTRL = int(0b0)
	# dut.RSTD = int(0b0)
	# dut.RSTINMODE = int(0b0)
	# dut.RSTM = int(0b0)
	# dut.RSTP = int(0b0)


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
	dut.data_in_A = 4
	dut.coefficient_in = 4
	yield clock_rising
	yield clock_rising
	dut.data_in_D = 4
	yield clock_rising
	yield clock_rising
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
	dut._log.info("data_in_carry = %d" % dut.data_in_carry.value.integer)