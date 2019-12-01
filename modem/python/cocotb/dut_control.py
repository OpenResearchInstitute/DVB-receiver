import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave

import numpy as np


class dut_control:
	"""
		A class which contains variety of useful and reused functions and 
		capabilities to control a DUT
	"""

	def __init__(self, dut):
		"""
			Initialise the object
		"""

		# save the DUT object internally
		self.dut = dut

		# default parameters
		self.CLK_PERIOD = 10


	@cocotb.coroutine
	def init_ports(self):
		"""
			Set any port initial values.
			This function is meant to be overrode by the child
		"""
		self.dut._log.info("The init_ports() function has not been overridden and therefore no action has been taken")
		yield Timer(0)




	@cocotb.coroutine
	def reset(self):
		"""
			Reset the DUT.
			This function is meant to be overrode by the child
		"""
		self.dut._log.info("The reset() function has not been overridden and therefore no action has been taken")
		yield Timer(0)




	@cocotb.coroutine
	def clock_start(self):
		"""
			Startup the clock required for the DUT.
			This function is meant to be overrode by the child
		"""
		self.dut._log.info("The clock_start() function has not been overridden and therefore no action has been taken")
		yield Timer(0)



	@cocotb.coroutine
	def setup_interfaces(self):
		"""
			Setup the DUT interfaces.
			This function is meant to be overrode by the child
		"""
		self.dut._log.info("The setup_interfaces() function has not been overridden and therefore no action has been taken")
		yield Timer(0)



	@cocotb.coroutine
	def init(self):
		"""
			Initialise the DUT
		"""

		# log the process
		self.dut._log.info("-"*64)
		self.dut._log.info("INITIALISING THE DUT")
		self.dut._log.info("")

		# initialse values
		self.dut._log.info("Setting initial port values")
		yield self.init_ports()

		# set the clock
		self.dut._log.info("Starting the clock")
		yield self.clock_start()

		# set the clock
		self.dut._log.info("Setting up interfaces")
		yield self.setup_interfaces()

		# reset the DUT
		self.dut._log.info("Resetting the DUT")
		yield self.reset()

		self.dut._log.info("DUT Initialisation complete")
		yield Timer(0)


	def print_elements(self):
		"""
			Traverses the dut printing out information about every element in
			the design
		"""

		# loop through the top level elements
		for design_element in self.dut:

			print("-"*100)
			print("Found %s : python type = %s: " % (design_element, type(design_element)))
			print("         : _name = %s: ", design_element._name)
			print("         : _path = %s: ", design_element._path)

			# found a sub element to push into
			if type(design_element) == cocotb.handle.HierarchyArrayObject or type(design_element) == cocotb.handle.HierarchyObject:
				print("Pushing into design element")
				print_elements(design_element)


	def GSR_control(self, dut, value):
		"""
			Allows the setting of the GSR (Global Set/Reset) in Xilinx unisim 
			elements.  Will traverse the whole design looking for a GSR signal name
			and will set it.
		"""

		# loop through the top level elements
		for design_element in dut:

			# check that the GSR signal has been found
			if design_element._name == 'GSR':
				design_element.value = value

			# found a sub element to push into
			if type(design_element) == cocotb.handle.HierarchyArrayObject or type(design_element) == cocotb.handle.HierarchyObject:
				self.GSR_control(design_element, value)


	def convert_to_signed(self, signal, number_bits):
		"""
			Convert the list of input numbers to python signed integer format for
			a given number of bits used to represent the number.
		"""

		output_signal = []

		for sample in signal:
			if sample > (2**(number_bits-1)-1):
				output_signal.append( sample - 2**number_bits )
			else:
				output_signal.append( sample )

		return output_signal


	@cocotb.coroutine
	def clk_wait(self, clk_rising, wait_period):
		"""
			Wait for a given number of clock cycles to be more compact for long
			clock delays
		"""

		for i in range(int(wait_period)):
			yield clk_rising


	def signed_to_fixedpoint(self, numbers, number_width, normalised=False, multiplier=1.0):
		"""
			Convert signed numbers to fixed point two's complement version
		"""

		# normalisation factor calculation
		if normalised:
			norm_factor = (2**(number_width-1)-1)/max(numbers)
		else:
			norm_factor = 1

		# support both single numbers and lists
		if type(numbers) == list:
			output = []
			for i, number in enumerate(numbers):
					if number < 0:
						output.append( int(np.round(2**(number_width) + multiplier*number*norm_factor)) )
					else:
						output.append( int(np.round(multiplier*number*norm_factor)) )
			return output

		else:
			if numbers < 0:
				output = int(np.round(2**(number_width) + number*norm_factor))
			else:
				output = int(np.round(number*norm_factor))
			return output



	def fixedpoint_to_signed(self, numbers, number_width, normalised=False):
		"""
			Convert fixed point two's complement number to signed number
		"""



		# support both single numbers and lists
		if type(numbers) == list:
			output = []
			for i, number in enumerate(numbers):
					if number > 2**(number_width-1)-1:
						output.append( (number - 2**number_width) )
					else:
						output.append( number )
		else:
			if numbers > 2**(number_width/2)-1:
				output = (numbers - 2**number_width)
			else:
				output = numbers


		# normalisation factor calculation
		if normalised:
			norm_factor = 2**number_width-1
		else:
			norm_factor = 1
		

		return [_/norm_factor for _ in output]
