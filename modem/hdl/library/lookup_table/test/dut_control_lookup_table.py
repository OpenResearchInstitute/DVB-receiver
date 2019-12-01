import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.triggers import NextTimeStep
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave

import sys
sys.path.insert(0, '../../../../python/cocotb')
from dut_control import dut_control
import helper_functions

import matplotlib.pyplot as plt


class dut_control_lookup_table(dut_control):
	"""
		A class which contains variety of useful and reused functions and 
		capabilities to control a DUT
	"""

	def __init__(self, dut):
		"""
			Initialise the object
		"""

		# perform parent initilisation first
		dut_control.__init__(self, dut)

		# set simulation parameters
		self.DEBUG = False
		self.PLOT = False

		# set system parameters
		self.DATA_WIDTH = 32
		self.MEMORY_DEPTH = 256



	@cocotb.coroutine
	def wait(self, wait_period):
		"""
			Wait for a given number of data in clock periods.
		"""
		yield helper_functions.clk_wait(self.data_in_clk_rising, wait_period)



	@cocotb.coroutine
	def reset(self):
		"""
			Reset the DUT.
		"""

		# reset the DUT
		self.dut.data_load_aresetn = 1
		self.dut.data_in_aresetn = 1
		self.dut.data_out_aresetn = 1
		yield self.wait(2)

		self.dut.data_load_aresetn = 0
		self.dut.data_in_aresetn = 0
		self.dut.data_out_aresetn = 0
		yield self.wait(2)

		self.dut.data_load_aresetn = 1
		self.dut.data_in_aresetn = 1
		self.dut.data_out_aresetn = 1
		helper_functions.GSR_control(self.dut, 0)
		yield self.wait(2)



	@cocotb.coroutine
	def clock_start(self):
		"""
			Startup the clock required for the DUT.
		"""
		self.dut._log.info("No independant clock used in design")

		yield Timer(0)


	@cocotb.coroutine
	def setup_interfaces(self):
		"""
			Setup the DUT interfaces.
		"""
		
		# input data interface
		self.data_in_clk_gen = cocotb.fork(Clock(self.dut.data_in_aclk, self.CLK_PERIOD).start())
		self.axism_data_in = AXI4StreamMaster(self.dut, "data_in", self.dut.data_in_aclk)

		# output data interface
		self.data_out_clk_gen = cocotb.fork(Clock(self.dut.data_out_aclk, self.CLK_PERIOD).start())
		self.axiss_data_out = AXI4StreamSlave(self.dut, "data_out", self.dut.data_out_aclk)

		# data load interface
		self.data_load_clk_gen = cocotb.fork(Clock(self.dut.data_load_aclk, self.CLK_PERIOD).start())
		self.axism_data_load = AXI4StreamMaster(self.dut, "data_load", self.dut.data_load_aclk)

		# use the input data clock
		self.data_in_clk_rising = yield RisingEdge(self.dut.data_in_aclk)

		yield Timer(0)



	@cocotb.coroutine
	def init_ports(self):
		"""
			Set any port initial values.
		"""

		# initialse values
		self.dut.data_in_aclk = 0
		self.dut.data_in_tdata = 0
		self.dut.data_in_tlast = 0
		self.dut.data_in_tvalid = 0

		yield Timer(0)



	@cocotb.coroutine
	def memory_write(self, data, convert_signed_to_fixed=True):
		"""
			Write into the DUT memory.
		"""

		# if requested plot the filter coefficients
		if self.PLOT:
			plt.plot(data)
			plt.title("Memory Input")
			plt.xlabel("Address")
			plt.ylabel("Value")
			plt.show()

		# convert negative numbers to twos complement and arrange for the polyphase structure
		if convert_signed_to_fixed:
			data = helper_functions.signed_to_fixedpoint(data, self.MEMORY_DEPTH)

		# write the coefficients through the bus
		yield self.axism_data_load.write(data)
		yield self.wait(4)



	@cocotb.coroutine
	def data_out_read_enable(self):
		"""
			Setup the handle to capture writes into the output data bus.
		"""

		self.axiss_read_handle = cocotb.fork(self.axiss_data_out.read())
		yield Timer(0)
		# yield self.wait(8)



	@cocotb.coroutine
	def memory_read(self, address=-1):
		"""
			Read the memory by altering the input address.
		"""

		# check if a particular memory address is requested
		if address == -1:

			# loop through each memory address
			address = [int(_) for _ in range(self.MEMORY_DEPTH)]

		# request value
		yield self.data_out_read_enable()

		# request data
		yield self.axism_data_in.write(address)

		# read out the values
		yield self.axiss_read_handle.join()
		read_values = self.axiss_data_out.data

		return read_values


