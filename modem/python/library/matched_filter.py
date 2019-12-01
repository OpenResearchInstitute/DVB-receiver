import numpy as np
from fir_filter import fir_filter
import gauss_pulse
import comms_filters

class matched_filter(fir_filter):

	def __init__(self, filter_type, samples_per_symbol, pulse_factor, pulse_length):
		""" Create the generic modulator object and specify the modulation parameters """

		fir_filter.__init__(self, coeffs = [0 + 1j*0], complex = True)

		# save the input parameters internally
		self.filter_type = filter_type
		self.samples_per_symbol = samples_per_symbol
		self.pulse_factor = pulse_factor
		self.pulse_length = pulse_length

		# create the pulse coefficients
		if(self.filter_type == "Gaussian"):
			# print("TODO: need to implement Laurent AMP decomposition for Gaussian pulse in Python!")
			self.pulse_coefficients = gauss_pulse.gauss_laurent_amp(sps = self.samples_per_symbol, 
																	BT = self.pulse_factor)
		else:
			self.pulse_coefficients = comms_filters.rrcosfilter(N = self.pulse_length*self.samples_per_symbol, 
																alpha  = self.pulse_factor, 
																Ts = 1, 
																Fs = self.samples_per_symbol)[1]

			self.pulse_coefficients = np.append(self.pulse_coefficients, self.pulse_coefficients[0])

		# normalise the pulse energy
		pulse_energy = np.sum(np.square(abs(self.pulse_coefficients)))/2.0
		self.pulse_coefficients = [_/pulse_energy for _ in self.pulse_coefficients]

		# initialse the underlying FIR filter now that the parameters are set
		self.update_coeffs(coeffs = self.pulse_coefficients, complex = True)