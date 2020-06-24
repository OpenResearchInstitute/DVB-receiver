import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure
from cocotb.drivers.amba import AXI4LiteMaster
from cocotb.drivers.amba import AXI4StreamMaster
from cocotb.drivers.amba import AXI4StreamSlave

import sys, json
sys.path.insert(0, '../../../../python/cocotb')
from dut_control import dut_control

sys.path.insert(0, '../../../../python/library')
import comms_filters
import generic_modem

import numpy as np
import matplotlib.pyplot as plt


class dut_control_apsk_modulator(dut_control):
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
		self.NUMBER_REPEAT_TESTS = 3

		# set system parameters
		self.INPUT_WIDTH = 32
		self.NUMBER_TAPS = 40
		self.SAMPLES_PER_SYMBOL = 4
		self.DATA_WIDTH = 16
		self.COEFFS_WIDTH = 16
		self.MEMORY_DEPTH = 256

		# test parameters
		self.POWER_TOLERANCE = 0.5
		self.SAMPLE_TOLERANCE = 1000.0/(2**self.DATA_WIDTH)
		self.LENGTH_TOLERANCE = 4

		# read in the modulation deifnition file
		self.modulation_definition_filename = "../../../../python/library/DVB-S2X_constellations.json"
		with open(self.modulation_definition_filename) as json_file:
			self.modulation_definition = json.load(json_file)



	@cocotb.coroutine
	def wait(self, wait_period):
		"""
			Wait for a given number of data in clock periods.
		"""
		yield self.clk_wait(self.data_in_clk_rising, wait_period)



	@cocotb.coroutine
	def reset(self):
		"""
			Reset the DUT.
		"""

		# reset the DUT
		self.dut.data_in_aresetn = 0
		self.dut.data_out_aresetn = 0
		self.dut.coefficients_in_aresetn = 0
		self.dut.control_aresetn = 0
		yield self.wait(2)

		self.dut.data_in_aresetn = 1
		self.dut.data_out_aresetn = 1
		self.dut.coefficients_in_aresetn = 1
		self.dut.control_aresetn = 1
		self.GSR_control(self.dut, 0)
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

		# input data interface
		self.lut_data_load_clk_gen = cocotb.fork(Clock(self.dut.lut_data_load_aclk, self.CLK_PERIOD).start())
		self.axism_lut_data_load = AXI4StreamMaster(self.dut, "lut_data_load", self.dut.lut_data_load_aclk)


		# coefficients interface
		self.coefficients_in_clk_gen = cocotb.fork(Clock(self.dut.coefficients_in_aclk, self.CLK_PERIOD).start())
		self.axism_coeffs_in = AXI4StreamMaster(self.dut, "coefficients_in", self.dut.coefficients_in_aclk)

		# control data interface
		self.control_clk_gen = cocotb.fork(Clock(self.dut.control_aclk, self.CLK_PERIOD).start())
		self.axilm_control = AXI4LiteMaster(self.dut, "control", self.dut.control_aclk)

		# use the input data clock
		self.data_in_clk_rising = yield RisingEdge(self.dut.data_in_aclk)

		yield Timer(0)



	@cocotb.coroutine
	def init_ports(self):
		"""
			Set any port initial values.
		"""

		# initialse values
		self.dut.data_in_tdata = 0
		self.dut.data_in_tlast = 0
		self.dut.data_in_tvalid = 0
		self.dut.data_out_tdata = 0
		self.dut.data_out_tlast = 0
		self.dut.data_out_tvalid = 0
		self.dut.lut_data_load_tdata = 0
		self.dut.lut_data_load_tlast = 0
		self.dut.lut_data_load_tvalid = 0
		self.dut.coefficients_in_tdata = 0
		self.dut.coefficients_in_tlast = 0
		self.dut.coefficients_in_tvalid = 0

		yield Timer(0)



	@cocotb.coroutine
	def set_modulation(self, modulation_name):
		"""
			Setup the DUT with the selected modulation type.
		"""

		# store the modulation to internal variable
		self.modulation_name = modulation_name

		# pull out the modulation parameters and store internally
		modulation_dict = self.modulation_definition[self.modulation_name[0]][self.modulation_name[1]]
		self.bits_per_symbol = modulation_dict['bits_per_symbol']
		self.pulse_filter_type = modulation_dict['filter']
		self.constellation_map = modulation_dict['bit_map']
		self.relative_rate = modulation_dict['relative_rate']
		self.symbol_offset = modulation_dict['offset']


		# TODO for now this is hardcoded here - should change
		#  should this be stored in the modulation JSON file?
		self.pulse_factor = 0.5


		# write in the coefficients
		yield self.coefficients_write()

		# write in the input data
		yield self.modulation_write()

		# write the control signals
		yield self.control_signals_write()



	@cocotb.coroutine
	def coefficients_write(self):
		"""
			Write coefficients into the DUT.
		"""

		# create the coefficients
		if self.pulse_filter_type == "RRC":
			coefficients = comms_filters.rrcosfilter(	N = self.NUMBER_TAPS, 
														alpha  = self.pulse_factor, 
														Ts = 1, 
														Fs = self.SAMPLES_PER_SYMBOL)[1]
		else:
			assert False, "Unsupported filter type (%s) specified" % self.pulse_filter_type

		# scale the coefficients 
		coefficients_max = sum(coefficients)
		coefficients = [int(coefficient * ((2**(self.COEFFS_WIDTH-1)-1)/coefficients_max)) for coefficient in coefficients]

		# if requested plot the filter coefficients
		if self.PLOT:
			plt.plot(coefficients)
			plt.title("Filter Coefficients")
			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()

		# convert negative numbers to twos complement and arrange for the polyphase structure
		coefficients = self.signed_to_fixedpoint(coefficients, self.COEFFS_WIDTH, multiplier=1.0)

		print(coefficients)

		# reset the coefficients
		self.dut.coefficients_in_aresetn = 0
		yield self.wait(2) 
		self.dut.coefficients_in_aresetn = 1
		yield self.wait(2) 

		# write the coefficients through the bus
		yield self.axism_coeffs_in.write(coefficients)
		yield self.wait(1)



	@cocotb.coroutine
	def modulation_write(self):
		"""
			Write modulation constellation the DUT memory.
		"""

		# set an amplitude below one
		amplitude = 2**14-1
		# amplitude = 1.0

		# convert negative numbers to twos complement and arrange for the polyphase structure
		i_constellation_map = [_[0] for _ in self.constellation_map]
		q_constellation_map = [_[1] for _ in self.constellation_map]

		# convert the signed floats to two's complement fixed point form		
		i_data = self.signed_to_fixedpoint(i_constellation_map, self.DATA_WIDTH, normalised=False, multiplier=amplitude)
		q_data = self.signed_to_fixedpoint(q_constellation_map, self.DATA_WIDTH, normalised=False, multiplier=amplitude)

		# combine the two constellation maps into a 32 bit number
		data = [int(q_data[i]*2**16) + int(i_data[i]) for i in range(len(i_data))]

		print(data)

		# reset the modulation
		self.dut.lut_data_load_aresetn = 0
		yield self.wait(2)
		self.dut.lut_data_load_aresetn = 1
		yield self.wait(2)

		# write the coefficients through the bus
		yield self.axism_lut_data_load.write(data)
		yield self.wait(1)



	@cocotb.coroutine
	def control_signals_write(self):
		"""
			Write the control signals to the DUT.
		"""

		# form the register as an integer
		register_value = self.symbol_offset*2**3 + self.bits_per_symbol
		
		# write the control signal
		yield self.axilm_control.write(address=0x00, value=register_value)
		yield self.wait(1)



	@cocotb.coroutine
	def data_out_read_enable(self):
		"""
			Setup the handle to capture writes into the output data bus.
		"""

		self.axiss_read_handle = cocotb.fork(self.axiss_data_out.read())
		yield self.wait(1)



	def data_out_read_parse(self):
		"""
			Parse the read data into real and imaginary parts.
		"""

		# split the data into real and imaginary parts
		data_read_real = [float((2**self.DATA_WIDTH-1) & int(_)) for _ in self.axiss_data_out.data]
		data_read_imag = [float(((2**self.DATA_WIDTH-1)*2**self.DATA_WIDTH) & int(_)) / 2**(self.DATA_WIDTH) for _ in self.axiss_data_out.data]


		data_read_real = self.fixedpoint_to_signed(data_read_real, self.DATA_WIDTH, normalised=True)
		data_read_imag = self.fixedpoint_to_signed(data_read_imag, self.DATA_WIDTH, normalised=True)


		# combine the data read into complex numbers
		multiplier = 5.67
		self.data_read = [data_read_real[i]*multiplier + 1j*data_read_imag[i]*multiplier for i in range(len(data_read_real))]



	@cocotb.coroutine
	def data_in_write(self, data):
		"""
			Write data to the modulator.
		"""

		# write in the data
		yield self.axism_data_in.write(data)



	def plot_constellation(self):
		"""
			Plot the received data on a constellation plot.
		"""

		# only display if the PLOT flag is enabled
		if self.PLOT:

			# split the output data into 
			real = [np.real(_) for _ in self.data_read]
			imag = [np.imag(_) for _ in self.data_read]

			# plot the constellation
			plt.scatter( real, imag )
			plt.title("Constellation")
			plt.xlabel("I")
			plt.ylabel("Q")
			plt.show()



	def plot_time_domain(self, series):
		"""
			Plot a time series.
		"""

		# only display if the PLOT flag is enabled
		if self.PLOT:

			if type(self.data_read[0]) == complex:

				# split the output data into 
				real = [np.real(_) for _ in self.data_read]
				imag = [np.imag(_) for _ in self.data_read]

				# plot the constellation
				plt.plot( real )
				plt.plot( imag )

			else:

				plt.plot( self.data_read )

			# label and render plot
			plt.title("Time Series")
			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()


	def test_modulation(self, input_data, tolerance=0.001):
		"""
			Test the received modulated data against the same data
			that was modulated using the Python implementation.
		"""

		# create the python modulator instance
		python_modulator = generic_modem.generic_modem(	modulation_type = self.modulation_name, 
														samples_per_symbol = self.SAMPLES_PER_SYMBOL, 
														pulse_factor = self.pulse_factor, 
														pulse_length = int(self.NUMBER_TAPS/self.SAMPLES_PER_SYMBOL), 
														filename = self.modulation_definition_filename)

		# covert the data to bits
		input_data_bits = []
		for number in input_data:
			for index in range(self.INPUT_WIDTH):
				input_data_bits.append( (number >> index) & 1 )

		# modulate the data with the python implemnentation
		expected = python_modulator.modulate(input_data_bits)
		received = self.data_read

		print(np.real(received))
		print(np.imag(received))

		# calculate the number of samples delay to propagate through pipelined registers
		delay = int((self.NUMBER_TAPS/self.SAMPLES_PER_SYMBOL-1)*self.SAMPLES_PER_SYMBOL)

		print(max([abs(_) for _ in expected]))
		print(max([abs(_) for _ in received]))

		# if requested plot the filter coefficients
		if self.PLOT:
			plt.subplot(211)
			plt.plot(np.real(expected))
			plt.plot(np.real(received))

			plt.subplot(212)
			plt.plot(np.imag(expected))
			plt.plot(np.imag(received))

			plt.legend(["Expected", "Received"])

			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()

		# check the lengths match
		if abs(len(expected) - len(received)) > self.LENGTH_TOLERANCE:
			raise TestFailure("The expected (len = %d) and received (len = %d) signals are of different length!" % (len(expected), len(received)))
		length = min(len(expected), len(received))

		# find the energies of both signals
		expected_energy = 0
		received_energy = 0
		for n in range(length):
			expected_energy += abs(expected[n])**2
			received_energy += abs(received[n])**2

		# convert to average power
		expected_power = expected_energy / len(expected)
		received_power = received_energy / len(received)

		# normalise the energies
		norm_factor = np.sqrt(expected_power/received_power)
		received = [_*norm_factor for _ in received]

		# if requested plot the filter coefficients
		if self.PLOT:
			plt.title("After Normalisation")
			plt.subplot(211)
			plt.plot(np.real(expected))
			plt.plot(np.real(received))

			plt.subplot(212)
			plt.plot(np.imag(expected))
			plt.plot(np.imag(received))

			plt.legend(["Expected", "Received"])

			plt.xlabel("Samples")
			plt.ylabel("Amplitude")
			plt.show()

		# find the power of the expected and received signal
		for n in range(length):
			difference = abs(expected[n] - received[n])
			if difference > self.SAMPLE_TOLERANCE:

				plt.subplot(211)
				plt.plot(np.real(expected))
				plt.plot(np.real(received))

				plt.subplot(212)
				plt.plot(np.imag(expected))
				plt.plot(np.imag(received))

				plt.legend(["Expected", "Received"])

				plt.xlabel("Samples")
				plt.ylabel("Amplitude")
				plt.show()

				
				raise TestFailure("The difference between the %d sample is %f which exceeds the tolerance of %f." % (n, difference, self.SAMPLE_TOLERANCE))