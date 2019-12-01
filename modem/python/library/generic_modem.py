import numpy as np
import fir_filter
import gauss_pulse
import comms_filters
import json

import matplotlib.pyplot as plt

class generic_modem:

	def __init__(	self, 
					modulation_type, 
					samples_per_symbol, 
					pulse_factor, 
					pulse_length, 
					filename,
					config=""	):
		""" 
			Create the generic MODEM (modulator / demodulator) object and specify 
			the modulation parameters.

			Parameters
			----------
			modulation_type : list
				Definition of the modulation type
			samples_per_symbol : int
				The number of samples to create per information symbol
			pulse_factor : float
				The pulse factor for either the RRC or Gaussian filters
			pulse_length : int
				The length of the pulse shaping filter in number of symbol periods
			filename : str
				The location of JSON file containing the modulation definition 
			config : optional
				Extra configuration setting
		""" 

		# save the input parameters internally
		self.modulation_type = modulation_type
		self.samples_per_symbol = samples_per_symbol
		self.pulse_factor = pulse_factor
		self.pulse_length = pulse_length
		self.config = config
		self.filename = filename

		# read in the modulation constellation definition
		with open(self.filename) as json_file:
			constellations = json.load(json_file)

		# pull out the modulation settings
		temp_dict = constellations
		for i in range(constellations["modulation_type_depth"]):
			temp_dict = temp_dict[self.modulation_type[i]]
		self.bits_per_symbol = temp_dict["bits_per_symbol"]
		self.offset = temp_dict["offset"]
		self.filter = temp_dict["filter"]
		self.relative_rate = temp_dict["relative_rate"]
		self.bit_map = temp_dict["bit_map"]

		# create the pulse coefficients
		if(self.filter == "Gaussian"):
			self.pulse_coefficients = gauss_pulse.gauss_pulse(	sps = 2*self.samples_per_symbol, 
																BT = self.pulse_factor)
		else:
			self.pulse_coefficients = comms_filters.rrcosfilter(N = self.pulse_length*self.samples_per_symbol, 
																alpha  = self.pulse_factor, 
																Ts = 1, 
																Fs = self.samples_per_symbol)[1]

		# normalise the pulse energy
		pulse_energy = np.sum(np.square(abs(self.pulse_coefficients)))/self.samples_per_symbol
		self.pulse_coefficients = [_/pulse_energy for _ in self.pulse_coefficients]
		self.pulse_coefficients = np.append(self.pulse_coefficients, self.pulse_coefficients[0])



	def modulate(self, data, carrier_phase_offset=0.0, empty_delay_line=True):
		""" 
			Modulate the supplied data with the previously setup modulator.

			Parameters
			----------
			data : list
				List of bits to modulate
			carrier_phase_offset : int, optional
				Initial carrier phase (default is 0.0)
		"""

		# determine the number of samples
		number_of_bits = len(data)
		number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/self.bits_per_symbol))

		# prepopulate the output vectors
		i_data = np.zeros(number_of_samples)
		q_data = np.zeros(number_of_samples)


		# loop through all data
		for n in range(int(np.floor(number_of_bits/self.bits_per_symbol))):
			
			# combine three bits and map to a complex symbol
			symbol_int = 0
			for i in range(self.bits_per_symbol):
				symbol_int += 2**i * data[self.bits_per_symbol*n + i]
			
			symbol = self.bit_map[symbol_int]

			# break apart the complex symbol to inphase and quadrature arms
			i_data[n*self.samples_per_symbol] = symbol[0]
			q_data[n*self.samples_per_symbol] = symbol[1]

		# if required append zeros to clear the delay line
		if empty_delay_line:
			i_data = np.concatenate((i_data, np.zeros(self.samples_per_symbol*(self.pulse_length-1))))
			q_data = np.concatenate((q_data, np.zeros(self.samples_per_symbol*(self.pulse_length-1))))


		# create the I and Q pulse filters
		i_filter = fir_filter.fir_filter(self.pulse_coefficients)
		q_filter = fir_filter.fir_filter(self.pulse_coefficients)

		# create output waveforms
		i_waveform = []
		q_waveform = []
		for n in range(len(i_data)):		    
		    i_waveform.append( i_filter.update( i_data[n] ) )
		    q_waveform.append( q_filter.update( q_data[n] ) )


		# create the complex signal and frequency offset
		waveform = [i_waveform[i] + 1j*q_waveform[i] for i in range(len(i_waveform))]
		waveform = [_*np.exp(-1j*carrier_phase_offset) for _ in waveform]
		    
		# normalise the waveform
		waveform_max = max( np.abs(waveform) )
		waveform = [_/waveform_max for _ in waveform]

		return waveform

