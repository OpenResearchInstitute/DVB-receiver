import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave
from cocotbext.axi import AxiLiteMaster

import sys
sys.path.insert(0, '../../../../python/cocotb')
from dut_control import dut_control
import helper_functions

import matplotlib.pyplot as plt


class dut_control_polyphase_filter(dut_control):
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
		# self.NUMBER_TAPS = 80
		self.NUMBER_TAPS = 32
		self.RATE_CHANGE = 8
		self.DATA_WIDTH = 16
		self.COEFFS_WIDTH = 16
		self.COEFF_STREAM = True
		self.COEFF_STREAM = False



	async def wait(self, wait_period):
		"""
			Wait for a given number of data in clock periods.
		"""
		await helper_functions.clk_wait(self.data_in_clk_rising, wait_period)



	async def reset(self):
		"""
			Reset the DUT.
		"""

		# reset the DUT
		self.dut.data_in_aresetn = 0
		self.dut.data_out_aresetn = 0
		if self.COEFF_STREAM:
			self.dut.coefficients_in_aresetn = 0
		else:
			self.dut.coeffs_axi_aresetn = 0
		await self.wait(2)

		self.dut.data_in_aresetn = 1
		self.dut.data_out_aresetn = 1
		if self.COEFF_STREAM:
			self.dut.coefficients_in_aresetn = 1
		else:
			self.dut.coeffs_axi_aresetn = 1
		helper_functions.GSR_control(self.dut, 0)
		await self.wait(2)



	async def clock_start(self):
		"""
			Startup the clock required for the DUT.
		"""
		self.dut._log.info("No independant clock used in design")

		await Timer(0)


	async def setup_interfaces(self):
		"""
			Setup the DUT interfaces.
		"""
		
		# coefficients interface
		if self.COEFF_STREAM:
			self.coefficients_in_clk_gen = cocotb.fork(Clock(self.dut.coefficients_in_aclk, self.CLK_PERIOD).start())
			self.axism_coeffs_in = AXI4StreamMaster(self.dut, "coefficients_in", self.dut.coefficients_in_aclk)
		else:
			self.coeffs_clk_gen = cocotb.fork(Clock(self.dut.coeffs_axi_aclk, self.CLK_PERIOD).start())
			self.axim_coeffs_in = AxiLiteMaster(self.dut, "coeffs_axi", self.dut.coeffs_axi_aclk)

		# input data interface
		self.data_in_clk_gen = cocotb.fork(Clock(self.dut.data_in_aclk, self.CLK_PERIOD).start())
		self.axism_data_in = AXI4StreamMaster(self.dut, "data_in", self.dut.data_in_aclk)

		# output data interface
		self.data_out_clk_gen = cocotb.fork(Clock(self.dut.data_out_aclk, self.CLK_PERIOD).start())
		self.axiss_data_out = AXI4StreamSlave(self.dut, "data_out", self.dut.data_out_aclk)

		# use the input data clock
		self.data_in_clk_rising = await RisingEdge(self.dut.data_out_aclk)

		await Timer(0)



	async def init_ports(self):
		"""
			Set any port initial values.
		"""

		# initialse values
		self.dut.data_in_aclk = 0
		self.dut.data_in_tdata = 0
		self.dut.data_in_tlast = 0
		self.dut.data_in_tvalid = 0

		await Timer(0)



	async def coefficients_write(self, coefficients, convert_signed_to_fixed=True):
		"""
			Write coefficients into the DUT.
		"""

		# if requested plot the filter coefficients
		if self.PLOT:
			plt.plot(coefficients)
			plt.title("Filter Coefficients")
			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()

		# convert negative numbers to twos complement and arrange for the polyphase structure
		if convert_signed_to_fixed:
			coefficients = helper_functions.signed_to_fixedpoint(coefficients, self.COEFFS_WIDTH)

		if self.COEFF_STREAM:

			# write the coefficients through the bus
			await self.axism_coeffs_in.write(coefficients)
			await self.wait(4)

		else:

			for coefficient in coefficients:
				print('coefficient', coefficient)
				# await self.axim_coeffs_in.write_dword(0x0000, coefficient, byteorder='big')
				await self.axim_coeffs_in.write_dword(0x0000, coefficient)


	async def data_out_read_enable(self):
		"""
			Setup the handle to capture writes into the output data bus.
		"""

		self.axiss_read_handle = cocotb.fork(self.axiss_data_out.read())
		await self.wait(8)



	async def impulse_response(self, expected_response):
		"""
			Find the impulse response of the DUT
		"""

		# send in a pulse
		self.dut._log.info("-"*64)
		self.dut._log.info("STARTING IMPULSE RESPONSE TEST")
		self.dut._log.info("")
		self.dut._log.info("Sending impulse...")

		# create an pulse input with enough trailing zeros to clear the delay line
		#  due to non-idealities in the HDL we need to add an extra zero
		# await self.axism_data_in.write([(2**(self.DATA_WIDTH-2))] + [0]*int((self.NUMBER_TAPS/self.RATE_CHANGE)+1))
		# await self.axism_data_in.write([(2**(self.DATA_WIDTH-2))] + [0]*2)
		# await self.axism_data_in.write([0]*2 + [(2**(self.DATA_WIDTH-2))])
		await self.axism_data_in.write([(2**(self.DATA_WIDTH-2))])

		# read the output
		await self.axiss_read_handle.join()
		received_data = helper_functions.fixedpoint_to_signed(self.axiss_data_out.data, self.COEFFS_WIDTH)

		# because of non-idealities in the clearing of the filter an extra zero was added, remove this
		#  on the output after it's been interpolated
		# received_data = received_data[:-self.RATE_CHANGE]

		# print or plot the results
		if self.DEBUG:
			print("Received data: ", received_data)
		if self.PLOT:
			plt.plot( received_data )
			plt.title("Received Signal")
			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()
		
		# test that the output is correct
		expected_output = expected_response
		if expected_output != received_data:
			print("Expected: ", expected_output)
			print("Received: ", received_data)
			raise TestFailure("Impulse response does not match the expected response!")
		else:
			self.dut._log.info("Correct impulse response received")


		await self.wait(16)