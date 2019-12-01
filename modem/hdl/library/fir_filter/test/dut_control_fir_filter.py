import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave

import sys
sys.path.insert(0, '../../../../python/cocotb')
from dut_control import dut_control
import helper_functions

class dut_control_fir_filter(dut_control):
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
		self.PLOT = True

		# set system parameters
		self.NUMBER_TAPS = 16
		self.DATA_WIDTH = 16
		self.COEFFS_WIDTH = 16



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
		self.dut.data_in_aresetn = 0
		self.dut.data_out_aresetn = 0
		self.dut.coefficients_in_aresetn = 0
		yield self.wait(2)

		self.dut.data_in_aresetn = 1
		self.dut.data_out_aresetn = 1
		self.dut.coefficients_in_aresetn = 1
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
		
		# coefficients interface
		self.coefficients_in_clk_gen = cocotb.fork(Clock(self.dut.coefficients_in_aclk, self.CLK_PERIOD).start())
		self.axism_coeffs_in = AXI4StreamMaster(self.dut, "coefficients_in", self.dut.coefficients_in_aclk)

		# input data interface
		self.data_in_clk_gen = cocotb.fork(Clock(self.dut.data_in_aclk, self.CLK_PERIOD).start())
		self.axism_data_in = AXI4StreamMaster(self.dut, "data_in", self.dut.data_in_aclk)

		# output data interface
		self.data_out_clk_gen = cocotb.fork(Clock(self.dut.data_out_aclk, self.CLK_PERIOD).start())
		self.axiss_data_out = AXI4StreamSlave(self.dut, "data_out", self.dut.data_out_aclk)

		# use the input data clock
		self.data_in_clk_rising = yield RisingEdge(self.dut.data_out_aclk)

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
	def coefficients_write(self, coefficients):
		"""
			Write coefficients into the DUT.
		"""

		yield self.axism_coeffs_in.write(coefficients)
		yield self.wait(4)



	@cocotb.coroutine
	def data_out_read_enable(self):
		"""
			Setup the handle to capture writes into the output data bus.
		"""

		self.axiss_read_handle = cocotb.fork(self.axiss_data_out.read())
		yield self.wait(8)



	@cocotb.coroutine
	def impulse_response(self, expected_response):
		"""
			Find the impulse response of the DUT
		"""

		# send in a pulse
		self.dut._log.info("-"*64)
		self.dut._log.info("STARTING IMPULSE RESPONSE TEST")
		self.dut._log.info("")
		self.dut._log.info("Sending impulse...")

		# write in a pulse with enough zeros to get all the coefficients
		# yield self.axism_data_in.write([(2**(self.DATA_WIDTH-2))] + [0]*(self.NUMBER_TAPS-1))
		yield self.axism_data_in.write([(2**(self.DATA_WIDTH-2))])

		# read the output
		yield self.axiss_read_handle.join()
		received_data = helper_functions.fixedpoint_to_signed(self.axiss_data_out.data, self.DATA_WIDTH)
		if self.DEBUG:
			print("Received data: ", received_data)
		
		# test that the output is correct
		expected_output = expected_response
		if expected_output != received_data[:len(expected_output)]:
			print("Expected: ", expected_output)
			print("Received: ", received_data[:len(expected_output)])
			raise TestFailure("Impulse response does not match the expected response!")
		else:
			self.dut._log.info("Correct impulse response received")
