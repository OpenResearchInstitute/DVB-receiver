import numpy as np
import fir_filter
import gauss_pulse
import comms_filters
import json

class modulator

	def __init__(self, modulation_type, samples_per_symbol, pulse_factor, pulse_length, config, filename):
		""" 
			Create the generic modulator object and specify the modulation parameters.
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

    	# select the modulation scheme
    	self.bits_per_symbol = constellations[self.config[0]][self.config[1]]["bits_per_symbol"]
    	self.offset = constellations[self.config[0]][self.config[1]]["offset"]
    	self.filter = constellations[self.config[0]][self.config[1]]["filter"]
    	self.relative_rate = constellations[self.config[0]][self.config[1]]["relative_rate"]
    	self.bit_map = constellations[self.config[0]][self.config[1]]["bit_map"]

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



	def modulate(self, data, carrier_phase_offset):
		""" 
			Modulate the supplied data with the previously setup modulator
		"""

		# determine the number of samples
		number_of_bits = len(data)
		number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/self.bits_per_symbol))

		# prepopulate the output vectors
		i_data = np.zeros(number_of_samples)
		q_data = np.zeros(number_of_samples)


		# loop through all data
		for n in range(int(np.ceil(number_of_bits/self.bits_per_symbol))):
			
			# combine three bits and map to a complex symbol
			symbol_int = 0
			for i in range(self.bits_per_symbol):
				symbol_int += 2**i * data[self.bits_per_symbol*n + i]

			symbol = self.bit_map[symbol_int]

			# break apart the complex symbol to inphase and quadrature arms
			i_data[n*self.samples_per_symbol] = np.real(symbol)
			q_data[n*self.samples_per_symbol] = np.imag(symbol)


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

