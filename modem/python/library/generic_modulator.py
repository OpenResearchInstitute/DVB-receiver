import numpy as np
import fir_filter
import gauss_pulse
import comms_filters

class generic_modulator:

	def __init__(self, modulation_type, samples_per_symbol, pulse_factor, pulse_length, config):
		""" Create the generic modulator object and specify the modulation parameters.
			Supported modulation types are:
				BPSK
				GMSK
				QPSK
				OQPSK
				8PSK
				8APSK
				16APSK
				32APSK
				64APSK
				128APSK 
				256APSK
		""" 

		# save the input parameters internally
		self.modulation_type = modulation_type
		self.samples_per_symbol = samples_per_symbol
		self.pulse_factor = pulse_factor
		self.pulse_length = pulse_length
		self.config = config

		# set the spectral density and offset characteristics
		if self.modulation_type == "BPSK":
			self.spectral_density = 2
			self.period_offset = 0
		elif self.modulation_type == "GMSK":
			self.spectral_density = 2
			self.period_offset = 1
		elif self.modulation_type == "QPSK":
			self.spectral_density = 4
			self.period_offset = 0
		elif self.modulation_type == "OQPSK":
			self.spectral_density = 4
			self.period_offset = 1
		elif self.modulation_type == "8PSK":
			self.spectral_density = 8
			self.period_offset = 0
		elif self.modulation_type == "8APSK":
			self.spectral_density = 8
			self.period_offset = 0
		elif self.modulation_type == "16APSK":
			self.spectral_density = 16
			self.period_offset = 0
		elif self.modulation_type == "32APSK":
			self.spectral_density = 32
			self.period_offset = 0
		elif self.modulation_type == "64APSK":
			self.spectral_density = 64
			self.period_offset = 0
		elif self.modulation_type == "128APSK":
			self.spectral_density = 128
			self.period_offset = 0
		elif self.modulation_type == "256APSK":
			self.spectral_density = 256
			self.period_offset = 0
		else:
			assert False, "Unsupported modulation type supplied."

		# create the pulse coefficients
		if(self.modulation_type == "GMSK"):
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
		""" Modulate the supplied data with the previously setup modulator """

		# deinterleave, convert to  NRZ and interpolate
		if self.modulation_type == "BPSK":

			# determine the number of samples
			number_of_bits = len(data)
			number_of_samples = number_of_bits*self.samples_per_symbol

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# loop through each sample modulate the in-phase arm
			for n in range(number_of_bits):
				i_data[n*self.samples_per_symbol] = 2*data[n]-1

			# the quadrature arm is all zeros
			q_data = np.zeros(number_of_samples)


		# essentially OQPSK with half the frequency
		elif self.modulation_type == "GMSK":

			# determine the number of samples
			number_of_bits = len(data)
			number_of_samples = number_of_bits*self.samples_per_symbol

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# modulate two bit period with data
			for n in range(number_of_bits/2):
				i_data[2*n*self.samples_per_symbol] = 2*data[2*n]-1

			# module two bit periods offset by a bit period with data
			for n in range(number_of_bits/2-1):
				q_data[2*n*self.samples_per_symbol + self.samples_per_symbol/2] = 2*data[2*n+1]-1


		# map the signal to four constellation points on the complex plane
		elif self.modulation_type == "QPSK":

			# determine the number of samples
			number_of_bits = len(data)
			number_of_samples = number_of_bits*self.samples_per_symbol/2

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# map every odd bit to the in-phase arm
			for n in range(number_of_bits/2):
				i_data[n*self.samples_per_symbol] = 2*data[2*n]-1

			# map every even bit to the quadarature arm
			for n in range(number_of_bits/2):
				q_data[n*self.samples_per_symbol] = 2*data[2*n+1]-1


		# like QPSK with a half bit period offset on the quadarature arm 
		elif self.modulation_type == "OQPSK":

			# determine the number of samples
			number_of_bits = len(data)
			number_of_samples = number_of_bits*self.samples_per_symbol/2

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# map every odd bit to the in-phase arm
			for n in range(number_of_bits/2):
				i_data[n*self.samples_per_symbol] = 2*data[2*n]-1

			# map every even bit to the quadarature arm with a half bit period offset
			for n in range(number_of_bits/2-1):
				q_data[n*self.samples_per_symbol + self.samples_per_symbol/2] = 2*data[2*n+1]-1


		# split three bits across a even eight point on the circle
		#  according to EN 302 307-1
		elif self.modulation_type == "8PSK":

			# determine the number of samples
			bits_per_symbol = 3
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# set the bit mapping table
			bit_map =  [[1, np.pi/4],
						[1, 0],
						[1, 4*np.pi/4],
						[1, 5*np.pi/4],
						[1, 2*np.pi/4],
						[1, 7*np.pi/4],
						[1, 3*np.pi/4],
						[1, 6*np.pi/4]]

						# loop through all data
			for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
				
				# combine three bits and map to a complex symbol
				symbol_int = 0
				for i in range(bits_per_symbol):
					symbol_int += 2**i * data[bits_per_symbol*n + i]

				symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

				# break apart the complex symbol to inphase and quadrature arms
				i_data[n*self.samples_per_symbol] = np.real(symbol)
				q_data[n*self.samples_per_symbol] = np.imag(symbol)


		# split three bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "8APSK":

			# determine the number of samples
			bits_per_symbol = 3
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vector
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# different mapping for different LDPC codes
			# calculate the symbol radiuses
			if self.config == "100/180":	
				R1 = 1.0/6.8
				R2 = 5.32/6.8

			elif self.config == "104/180":
				R1 = 1.0/8.0
				R2 = 6.39/8.0
			
			else:
				print("No LDPC code specified.  Using 100/180")
				R1 = 1.0/6.8
				R2 = 5.32/6.8			

			# set the bit mapping table
			bit_map =  [[R1, 0],
						[R2, 1.352*np.pi],
						[R2, 0.648*np.pi],
						[1.0, 0],
						[R1, np.pi],
						[R2, -0.352*np.pi],
						[R2, 0.352*np.pi],
						[1.0, np.pi]]

			# loop through all data
			for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
				
				# combine three bits and map to a complex symbol
				symbol_int = 0
				for i in range(bits_per_symbol):
					symbol_int += 2**i * data[bits_per_symbol*n + i]

				symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

				# break apart the complex symbol to inphase and quadrature arms
				i_data[n*self.samples_per_symbol] = np.real(symbol)
				q_data[n*self.samples_per_symbol] = np.imag(symbol)


		# split four bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "16APSK":

			# determine the number of samples
			bits_per_symbol = 4
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# for some codes the mapping is performed with a lookup table
			if self.config in ["18/30", "20/30"]:
			
				if self.config == "18/30":

					# map values to symbols on the complex plane
					bit_map =  [0.4718 + 0.2606*1j,
								0.2606 + 0.4718*1j,
								-0.4718 + 0.2606*1j,
								-0.2606 + 0.4718*1j,
								0.4718 - 0.2606*1j,
								0.2606 - 0.4718*1j,
								-0.4718 - 0.2606*1j,
								-0.2606 - 0.4718*1j,
								1.2088 + 0.4984*1j,
								0.4984 + 1.2088*1j,
								-1.2088 + 0.4984*1j,
								-0.4984 + 1.2088*1j,
								1.2088 - 0.4984*1j,
								0.4984 - 1.2088*1j,
								-1.2088 - 0.4984*1j,
								-0.4984 - 1.2088*1j]

				elif self.config == "20/30":

					# map values to symbols on the complex plane
					bit_map =  [0.5061 + 0.2474*1j,
								0.2474 + 0.5061*1j,
								-0.5061 + 0.2474*1j,
								-0.2474 + 0.5061*1j,
								0.5061 - 0.2474*1j,
								0.2474 - 0.5061*1j,
								-0.5061 - 0.2474*1j,
								-0.2474 - 0.5061*1j,
								1.2007 + 0.4909*1j,
								0.4909 + 1.2007*1j,
								-1.2007 + 0.4909*1j,
								-0.4909 + 1.2007*1j,
								1.2007 - 0.4909*1j,
								0.4909 - 1.2007*1j,
								-1.2007 - 0.4909*1j,
								-0.4909 - 1.2007*1j]

				# loop through all data
				for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
					
					# combine three bits and map to a complex symbol
					symbol_int = 0
					for i in range(bits_per_symbol):
						symbol_int += 2**i * data[bits_per_symbol*n + i]

					symbol = bit_map[symbol_int]

					# break apart the complex symbol to inphase and quadrature arms
					i_data[n*self.samples_per_symbol] = np.real(symbol)
					q_data[n*self.samples_per_symbol] = np.imag(symbol)

			else:

				# 8 + 8 modulation
				if self.config in ["90/180", "96/180", "100/180"]:

					# all of these codes use the same R1 radius
					R1 = 1.0/3.7

					# set the bit mapping table
					bit_map =  [[R1, 1*np.pi/8],
								[R1, 3*np.pi/8],
								[R1, 7*np.pi/8],
								[R1, 5*np.pi/8],
								[R1, 15*np.pi/8],
								[R1, 13*np.pi/8],
								[R1, 9*np.pi/8],
								[R1, 11*np.pi/8],
								[1.0, 1*np.pi/8],
								[1.0, 3*np.pi/8],
								[1.0, 7*np.pi/8],
								[1.0, 5*np.pi/8],
								[1.0, 15*np.pi/8],
								[1.0, 13*np.pi/8],
								[1.0, 9*np.pi/8],
								[1.0, 11*np.pi/8]]

				# 4 + 12 modulation
				else:

					# different mapping for different LDPC codes
					# calculate the symbol radiuses
					if self.config == "26/45":	
						R1 = 1.0/3.7
					elif self.config == "3/5":
						R1 = 1.0/3.7
					elif self.config == "28/45":
						R1 = 1.0/3.5
					elif self.config == "23/36":
						R1 = 1.0/3.1
					elif self.config == "25/36":
						R1 = 1.0/3.1
					elif self.config == "13/18":
						R1 = 1.0/2.85
					elif self.config == "140/180":
						R1 = 1.0/3.6
					elif self.config == "154/180":
						R1 = 1.0/3.2
					elif self.config == "7/15":
						R1 = 1.0/3.32
					elif self.config == "8/15":
						R1 = 1.0/3.5
					elif self.config == "26/45":
						R1 = 1.0/3.7
					elif self.config == "3/5":
						R1 = 1.0/3.7
					elif self.config == "32/45":
						R1 = 1.0/2.85
					else:
						print("No LDPC code specified.  Using 3/5")
						R1 = 1.0/3.7

					# set the bit mapping table
					bit_map =  [[1.0,	3*np.pi/12],
								[1.0,	21*np.pi/12],
								[1.0,	9*np.pi/12],
								[1.0,	15*np.pi/12],
								[1.0,	1*np.pi/12],
								[1.0,	23*np.pi/12],
								[1.0,	11*np.pi/12],
								[1.0,	13*np.pi/12],
								[1.0,	5*np.pi/12],
								[1.0,	19*np.pi/12],
								[1.0,	7*np.pi/12],
								[1.0,	17*np.pi/12],
								[R1, 	3*np.pi/12],
								[R1, 	21*np.pi/12],
								[R1, 	9*np.pi/12],
								[R1, 	15*np.pi/12]]

				# loop through all data
				for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
					
					# combine three bits and map to a complex symbol
					symbol_int = 0
					for i in range(bits_per_symbol):
						symbol_int += 2**i * data[bits_per_symbol*n + i]

					symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

					# break apart the complex symbol to inphase and quadrature arms
					i_data[n*self.samples_per_symbol] = np.real(symbol)
					q_data[n*self.samples_per_symbol] = np.imag(symbol)



		# split five bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "32APSK":

			# determine the number of samples
			bits_per_symbol = 5
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)


			if self.config in ["2/3", "2/3S", "32/45S"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				if self.config == "2/3":	
					R1 = 1.0/5.55
					R2 = 2.85/5.55
				elif self.config == "2/3S":
					R1 = 1.0/5.54
					R2 = 2.84/5.54
				elif self.config == "32/45S":
					R1 = 1.0/5.26
					R2 = 2.84/5.26

				# set the bit mapping table
				bit_map =  [[1,	11*np.pi/16],
							[1,	9*np.pi/16],
							[1,	5*np.pi/16],
							[1,	7*np.pi/16],
							[R2, 9*np.pi/12],
							[R2, 7*np.pi/12],
							[R2, 3*np.pi/12],
							[R2, 5*np.pi/12],
							[1,	13*np.pi/16],
							[1,	15*np.pi/16],
							[1,	3*np.pi/16],
							[1,	1*np.pi/16],
							[R2, 11*np.pi/12],
							[R1, 3*np.pi/4],
							[R2, 1*np.pi/12],
							[R1, 1*np.pi/4],
							[1,	21*np.pi/16],
							[1,	23*np.pi/16],
							[1,	27*np.pi/16],
							[1,	25*np.pi/16],
							[R2, 15*np.pi/12],
							[R2, 17*np.pi/12],
							[R2, 21*np.pi/12],
							[R2, 19*np.pi/12],
							[1,	19*np.pi/16],
							[1,	17*np.pi/16],
							[1,	29*np.pi/16],
							[1,	31*np.pi/16],
							[R2, 13*np.pi/12],
							[R1, 5*np.pi/4],
							[R2, 23*np.pi/12],
							[R1, 7*np.pi/4]]

			else:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				if self.config == "128/180":	
					R1 = 1.0/5.6
					R2 = 2.6/5.6
					R3 = 2.99/5.6
				elif self.config == "132/180":
					R1 = 1/5.6
					R2 = 2.6/5.6
					R3 = 2.86/5.6
				elif self.config == "140/180":
					R1 = 1/5.6
					R2 = 2.8/5.6
					R3 = 3.08/5.6
				else:
					print("No LDPC code specified.  Using 128/180")
					R1 = 1/5.6
					R2 = 2.6/5.6
					R3 = 2.99/5.6

				# set the bit mapping table
				bit_map =  [[R1, 1*np.pi/4],
							[1.0, 7*np.pi/16],
							[R1, 7*np.pi/4],
							[1.0, 25*np.pi/16],
							[R1, 3*np.pi/4],
							[1.0, 9*np.pi/16],
							[R1, 5*np.pi/4],
							[1.0, 23*np.pi/16],
							[R2, 1*np.pi/12],
							[1.0, 1*np.pi/16],
							[R2, 23*np.pi/12],
							[1.0, 31*np.pi/16],
							[R2, 11*np.pi/12],
							[1.0, 15*np.pi/16],
							[R2, 13*np.pi/12],
							[1.0, 17*np.pi/16],
							[R2, 5*np.pi/12],
							[1.0, 5*np.pi/16],
							[R2, 19*np.pi/12],
							[1.0, 27*np.pi/16],
							[R2, 7*np.pi/12],
							[1.0, 11*np.pi/16],
							[R2, 17*np.pi/12],
							[1.0, 21*np.pi/16],
							[R3, 1*np.pi/4],
							[1.0, 3*np.pi/16],
							[R3, 7*np.pi/4],
							[1.0, 29*np.pi/16],
							[R3, 3*np.pi/4],
							[1.0, 13*np.pi/16],
							[R3, 5*np.pi/4],
							[1.0, 19*np.pi/16]]


				# loop through all data
				for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
					
					# combine three bits and map to a complex symbol
					symbol_int = 0
					for i in range(bits_per_symbol):
						symbol_int += 2**i * data[bits_per_symbol*n + i]

					symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

					# break apart the complex symbol to inphase and quadrature arms
					i_data[n*self.samples_per_symbol] = np.real(symbol)
					q_data[n*self.samples_per_symbol] = np.imag(symbol)



		# split six bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "64APSK":

			# determine the number of samples
			bits_per_symbol = 6
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)


			if self.config in ["128/180"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				R1 = 1.0/3.95
				R2 = 1.88/3.95
				R3 = 2.72/3.95
				R4 = 1.0

				# set the bit mapping table
				bit_map =  [R1, 1*np.pi/16,
							R1, 3*np.pi/16,
							R1, 7*np.pi/16,
							R1, 5*np.pi/16,
							R1, 15*np.pi/16,
							R1, 13*np.pi/16,
							R1, 9*np.pi/16,
							R1, 11*np.pi/16,
							R1, 31*np.pi/16,
							R1, 29*np.pi/16,
							R1, 25*np.pi/16,
							R1, 27*np.pi/16,
							R1, 17*np.pi/16,
							R1, 19*np.pi/16,
							R1, 23*np.pi/16,
							R1, 21*np.pi/16,
							R2, 1*np.pi/16,
							R2, 3*np.pi/16,
							R2, 7*np.pi/16,
							R2, 5*np.pi/16,
							R2, 15*np.pi/16,
							R2, 13*np.pi/16,
							R2, 9*np.pi/16,
							R2, 11*np.pi/16,
							R2, 31*np.pi/16,
							R2, 29*np.pi/16,
							R2, 25*np.pi/16,
							R2, 27*np.pi/16,
							R2, 17*np.pi/16,
							R2, 19*np.pi/16,
							R2, 23*np.pi/16,
							R2, 21*np.pi/16,
							R4, 1*np.pi/16,
							R4, 3*np.pi/16,
							R4, 7*np.pi/16,
							R4, 5*np.pi/16,
							R4, 15*np.pi/16,
							R4, 13*np.pi/16,
							R4, 9*np.pi/16,
							R4, 11*np.pi/16,
							R4, 31*np.pi/16,
							R4, 29*np.pi/16,
							R4, 25*np.pi/16,
							R4, 27*np.pi/16,
							R4, 17*np.pi/16,
							R4, 19*np.pi/16,
							R4, 23*np.pi/16,
							R4, 21*np.pi/16,
							R3, 1*np.pi/16,
							R3, 3*np.pi/16,
							R3, 7*np.pi/16,
							R3, 5*np.pi/16,
							R3, 15*np.pi/16,
							R3, 13*np.pi/16,
							R3, 9*np.pi/16,
							R3, 11*np.pi/16,
							R3, 31*np.pi/16,
							R3, 29*np.pi/16,
							R3, 25*np.pi/16,
							R3, 27*np.pi/16,
							R3, 17*np.pi/16,
							R3, 19*np.pi/16,
							R3, 23*np.pi/16,
							R3, 21*np.pi/16]


			elif self.config in ["7/9", "4/5", "5/6"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				if self.config == "7/9":	
					R1 = 1.0/5.2
					R2 = 2.2/5.2
					R3 = 3.6/5.2
					R4 = 1.0
				elif self.config == "4/5":
					R1 = 1.0/5.2
					R2 = 2.2/5.2
					R3 = 3.6/5.2
					R4 = 1.0
				elif self.config == "5/6":
					R1 = 1.0/5.0
					R2 = 2.2/5.0
					R3 = 3.5/5.0
					R4 = 1.0

				# set the bit mapping table
				bit_map =  [R2, 25*np.pi/16,
							R4, 7*np.pi/4,
							R2, 27*np.pi/16,
							R3, 7*np.pi/4,
							R4, 31*np.pi/20,
							R4, 33*np.pi/20,
							R3, 31*np.pi/20,
							R3, 33*np.pi/20,
							R2, 23*np.pi/16,
							R4, 5*np.pi/4,
							R2, 21*np.pi/16,
							R3, 5*np.pi/4,
							R4, 29*np.pi/20,
							R4, 27*np.pi/20,
							R3, 29*np.pi/20,
							R3, 27*np.pi/20,
							R1, 13*np.pi/8,
							R4, 37*np.pi/20,
							R2, 29*np.pi/16,
							R3, 37*np.pi/20,
							R1, 15*np.pi/8,
							R4, 39*np.pi/20,
							R2, 31*np.pi/16,
							R3, 39*np.pi/20,
							R1, 11*np.pi/8,
							R4, 23*np.pi/20,
							R2, 19*np.pi/16,
							R3, 23*np.pi/20,
							R1, 9*np.pi/8,
							R4, 21*np.pi/20,
							R2, 17*np.pi/16,
							R3, 21*np.pi/20,
							R2, 7*np.pi/6,
							R4, 1*np.pi/4,
							R2, 5*np.pi/6,
							R3, 1*np.pi/4,
							R4, 9*np.pi/0,
							R4, 7*np.pi/0,
							R3, 9*np.pi/0,
							R3, 7*np.pi/0,
							R2, 9*np.pi/6,
							R4, 3*np.pi/4,
							R2, 11*np.pi/16,
							R3, 3*np.pi/4,
							R4, 11*np.pi/20,
							R4, 13*np.pi/20,
							R3, 11*np.pi/20,
							R3, 13*np.pi/20,
							R1, 3*np.pi/8,
							R4, 3*np.pi/0,
							R2, 3*np.pi/6,
							R3, 3*np.pi/0,
							R1, 1*np.pi/8,
							R4, 1*np.pi/0,
							R2, 1*np.pi/6,
							R3, 1*np.pi/0,
							R1, 5*np.pi/8,
							R4, 17*np.pi/20,
							R2, 13*np.pi/16,
							R3, 17*np.pi/20,
							R1, 7*np.pi/8,
							R4, 19*np.pi/20,
							R2, 15*np.pi/16,
							R3, 19*np.pi/20]


			elif self.config in ["132/180"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				R1 = 1.0/7.0
				R2 = 2.4/7.0
				R3 = 4.3/7.0
				R4 = 1.0

				# set the bit mapping table
				bit_map =  [R4, 1*np.pi/4,
							R4, 7*np.pi/4,
							R4, 3*np.pi/4,
							R4, 5*np.pi/4,
							R4, 13*np.pi/28,
							R4, 43*np.pi/28,
							R4, 15*np.pi/28,
							R4, 41*np.pi/28,
							R4, 1*np.pi/8,
							R4, 55*np.pi/28,
							R4, 27*np.pi/28,
							R4, 29*np.pi/28,
							R1, 1*np.pi/4,
							R1, 7*np.pi/4,
							R1, 3*np.pi/4,
							R1, 5*np.pi/4,
							R4, 9*np.pi/8,
							R4, 47*np.pi/28,
							R4, 19*np.pi/28,
							R4, 37*np.pi/28,
							R4, 11*np.pi/28,
							R4, 45*np.pi/28,
							R4, 17*np.pi/28,
							R4, 39*np.pi/28,
							R3, 1*np.pi/0,
							R3, 39*np.pi/20,
							R3, 19*np.pi/20,
							R3, 21*np.pi/20,
							R2, 1*np.pi/2,
							R2, 23*np.pi/12,
							R2, 11*np.pi/12,
							R2, 13*np.pi/12,
							R4, 5*np.pi/8,
							R4, 51*np.pi/28,
							R4, 23*np.pi/28,
							R4, 33*np.pi/28,
							R3, 9*np.pi/0,
							R3, 31*np.pi/20,
							R3, 11*np.pi/20,
							R3, 29*np.pi/20,
							R4, 3*np.pi/8,
							R4, 53*np.pi/28,
							R4, 25*np.pi/28,
							R4, 31*np.pi/28,
							R2, 9*np.pi/0,
							R2, 19*np.pi/12,
							R2, 7*np.pi/2,
							R2, 17*np.pi/12,
							R3, 1*np.pi/4,
							R3, 7*np.pi/4,
							R3, 3*np.pi/4,
							R3, 5*np.pi/4,
							R3, 7*np.pi/0,
							R3, 33*np.pi/20,
							R3, 13*np.pi/20,
							R3, 27*np.pi/20,
							R3, 3*np.pi/0,
							R3, 37*np.pi/20,
							R3, 17*np.pi/20,
							R3, 23*np.pi/20,
							R2, 1*np.pi/4,
							R2, 7*np.pi/4,
							R2, 3*np.pi/4,
							R2, 5*np.pi/4]

			# loop through all data
			for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
				
				# combine three bits and map to a complex symbol
				symbol_int = 0
				for i in range(bits_per_symbol):
					symbol_int += 2**i * data[bits_per_symbol*n + i]

				symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

				# break apart the complex symbol to inphase and quadrature arms
				i_data[n*self.samples_per_symbol] = np.real(symbol)
				q_data[n*self.samples_per_symbol] = np.imag(symbol)


		# split seven bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "128APSK":

			# determine the number of samples
			bits_per_symbol = 7
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# select the LDPC codes
			if self.config in ["135/180", "140/180"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				R1 = 1.0/3.819
				R2 = 1.715/3.819
				R3 = 2.118/3.819
				R4 = 2.681/3.819
				R5 = 2.75/3.819
				R6 = 1.0

				# set the bit mapping table
				bit_map =  [R1, 83*np.pi/60,
							R6, 11*np.pi/05,
							R6, 37*np.pi/80,
							R6, 11*np.pi/68,
							R2, 121*np.pi/520,
							R3, 23*np.pi/80,
							R5, 19*np.pi/20,
							R4, 61*np.pi/20,
							R1, 103*np.pi/560,
							R6, 61*np.pi/20,
							R6, 383*np.pi/680,
							R6, 61*np.pi/20,
							R2, 113*np.pi/560,
							R3, 169*np.pi/008,
							R5, 563*np.pi/520,
							R4, 139*np.pi/840,
							R1, 243*np.pi/560,
							R6, 1993*np.pi/5040,
							R6, 43*np.pi/90,
							R6, 73*np.pi/68,
							R2, 1139*np.pi/2520,
							R3, 117*np.pi/280,
							R5, 341*np.pi/720,
							R4, 349*np.pi/840,
							R1, 177*np.pi/560,
							R6, 1789*np.pi/5040,
							R6, 49*np.pi/80,
							R6, 1789*np.pi/5040,
							R2, 167*np.pi/560,
							R3, 239*np.pi/720,
							R5, 199*np.pi/720,
							R4, 281*np.pi/840,
							R1, 1177*np.pi/1260,
							R6, 94*np.pi/05,
							R6, 1643*np.pi/1680,
							R6, 157*np.pi/168,
							R2, 2399*np.pi/2520,
							R3, 257*np.pi/280,
							R5, 701*np.pi/720,
							R4, 659*np.pi/720,
							R1, 457*np.pi/560,
							R6, 359*np.pi/420,
							R6, 1297*np.pi/1680,
							R6, 4111*np.pi/5040,
							R2, 447*np.pi/560,
							R3, 839*np.pi/008,
							R5, 1957*np.pi/2520,
							R4, 701*np.pi/840,
							R1, 317*np.pi/560,
							R6, 3047*np.pi/5040,
							R6, 47*np.pi/90,
							R6, 95*np.pi/68,
							R2, 1381*np.pi/2520,
							R3, 163*np.pi/280,
							R5, 379*np.pi/720,
							R4, 491*np.pi/840,
							R1, 383*np.pi/560,
							R6, 3251*np.pi/5040,
							R6, 131*np.pi/180,
							R6, 115*np.pi/168,
							R2, 393*np.pi/560,
							R3, 481*np.pi/720,
							R5, 521*np.pi/720,
							R4, 559*np.pi/840,
							R1, 2437*np.pi/1260,
							R6, 199*np.pi/105,
							R6, 3323*np.pi/1680,
							R6, 325*np.pi/168,
							R2, 4919*np.pi/2520,
							R3, 537*np.pi/280,
							R5, 1421*np.pi/720,
							R4, 1379*np.pi/720,
							R1, 1017*np.pi/560,
							R6, 779*np.pi/420,
							R6, 2977*np.pi/1680,
							R6, 9151*np.pi/5040,
							R2, 1007*np.pi/560,
							R3, 1847*np.pi/1008,
							R5, 4477*np.pi/2520,
							R4, 1541*np.pi/840,
							R1, 877*np.pi/560,
							R6, 8087*np.pi/5040,
							R6, 137*np.pi/90,
							R6, 263*np.pi/168,
							R2, 3901*np.pi/2520,
							R3, 443*np.pi/280,
							R5, 1099*np.pi/720,
							R4, 1331*np.pi/840,
							R1, 943*np.pi/560,
							R6, 8291*np.pi/5040,
							R6, 311*np.pi/180,
							R6, 283*np.pi/168,
							R2, 953*np.pi/560,
							R3, 1201*np.pi/720,
							R5, 1241*np.pi/720,
							R4, 1399*np.pi/840,
							R1, 1343*np.pi/1260,
							R6, 116*np.pi/105,
							R6, 1717*np.pi/1680,
							R6, 179*np.pi/168,
							R2, 2641*np.pi/2520,
							R3, 303*np.pi/280,
							R5, 739*np.pi/720,
							R4, 781*np.pi/720,
							R1, 663*np.pi/560,
							R6, 481*np.pi/420,
							R6, 2063*np.pi/1680,
							R6, 5969*np.pi/5040,
							R2, 673*np.pi/560,
							R3, 1177*np.pi/1008,
							R5, 3083*np.pi/2520,
							R4, 979*np.pi/840,
							R1, 803*np.pi/560,
							R6, 7033*np.pi/5040,
							R6, 133*np.pi/90,
							R6, 241*np.pi/168,
							R2, 3659*np.pi/2520,
							R3, 397*np.pi/280,
							R5, 1061*np.pi/720,
							R4, 1189*np.pi/840,
							R1, 737*np.pi/560,
							R6, 6829*np.pi/5040,
							R6, 229*np.pi/180,
							R6, 221*np.pi/168,
							R2, 727*np.pi/560,
							R3, 959*np.pi/720,
							R5, 919*np.pi/720,
							R4, 1121*np.pi/840]

							# loop through all data
			for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
				
				# combine three bits and map to a complex symbol
				symbol_int = 0
				for i in range(bits_per_symbol):
					symbol_int += 2**i * data[bits_per_symbol*n + i]

				symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

				# break apart the complex symbol to inphase and quadrature arms
				i_data[n*self.samples_per_symbol] = np.real(symbol)
				q_data[n*self.samples_per_symbol] = np.imag(symbol)



		# split eight bits across a complex amplitudde and phase mapping
		elif self.modulation_type == "256APSK":

			# determine the number of samples
			bits_per_symbol = 8
			number_of_bits = len(data)
			number_of_samples = int(np.ceil(number_of_bits*self.samples_per_symbol/bits_per_symbol))

			# prepopulate the output vectors
			i_data = np.zeros(number_of_samples)
			q_data = np.zeros(number_of_samples)

			# select the coding based on the LDPC code
			if self.config in ["116/180", "124/180", "128/180", "135/180"]:

				# different mapping for different LDPC codes
				# calculate the symbol radiuses
				if self.config in ["116/180", "124/180"]:
					R1 = 1.0/6.536
					R2 = 1.791/6.536
					R3 = 2.405/6.536
					R4 = 2.980/6.536
					R5 = 3.569/6.536
					R6 = 4.235/6.536
					R7 = 5.078/6.536
					R8 = 1.0
				elif self.config == "128/180":
					R1 = 1.0/5.4
					R2 = 1.794/5.4
					R3 = 2.409/5.4
					R4 = 2.986/5.4
					R5 = 3.579/5.4
					R6 = 4.045/5.4
					R7 = 4.6/5.4
					R8 = 1.0
				else:
					R1 = 1.0/5.2
					R2 = 1.794/5.2
					R3 = 2.409/5.2
					R4 = 2.986/5.2
					R5 = 3.579/5.2
					R6 = 4.045/5.2
					R7 = 4.5/5.2
					R8 = 1.0

				# set the bit mapping table
				bit_map =  [R1, 1*np.pi/32,
							R1, 3*np.pi/32,
							R1, 7*np.pi/32,
							R1, 5*np.pi/32,
							R1, 15*np.pi/32,
							R1, 13*np.pi/32,
							R1, 9*np.pi/32,
							R1, 11*np.pi/32,
							R1, 31*np.pi/32,
							R1, 29*np.pi/32,
							R1, 25*np.pi/32,
							R1, 27*np.pi/32,
							R1, 17*np.pi/32,
							R1, 19*np.pi/32,
							R1, 23*np.pi/32,
							R1, 21*np.pi/32,
							R1, 63*np.pi/32,
							R1, 61*np.pi/32,
							R1, 57*np.pi/32,
							R1, 59*np.pi/32,
							R1, 49*np.pi/32,
							R1, 51*np.pi/32,
							R1, 55*np.pi/32,
							R1, 53*np.pi/32,
							R1, 33*np.pi/32,
							R1, 35*np.pi/32,
							R1, 39*np.pi/32,
							R1, 37*np.pi/32,
							R1, 47*np.pi/32,
							R1, 45*np.pi/32,
							R1, 41*np.pi/32,
							R1, 43*np.pi/32,
							R2, 1*np.pi/32,
							R2, 3*np.pi/32,
							R2, 7*np.pi/32,
							R2, 5*np.pi/32,
							R2, 15*np.pi/32,
							R2, 13*np.pi/32,
							R2, 9*np.pi/32,
							R2, 11*np.pi/32,
							R2, 31*np.pi/32,
							R2, 29*np.pi/32,
							R2, 25*np.pi/32,
							R2, 27*np.pi/32,
							R2, 17*np.pi/32,
							R2, 19*np.pi/32,
							R2, 23*np.pi/32,
							R2, 21*np.pi/32,
							R2, 63*np.pi/32,
							R2, 61*np.pi/32,
							R2, 57*np.pi/32,
							R2, 59*np.pi/32,
							R2, 49*np.pi/32,
							R2, 51*np.pi/32,
							R2, 55*np.pi/32,
							R2, 53*np.pi/32,
							R2, 33*np.pi/32,
							R2, 35*np.pi/32,
							R2, 39*np.pi/32,
							R2, 37*np.pi/32,
							R2, 47*np.pi/32,
							R2, 45*np.pi/32,
							R2, 41*np.pi/32,
							R2, 43*np.pi/32,
							R4, 1*np.pi/32,
							R4, 3*np.pi/32,
							R4, 7*np.pi/32,
							R4, 5*np.pi/32,
							R4, 15*np.pi/32,
							R4, 13*np.pi/32,
							R4, 9*np.pi/32,
							R4, 11*np.pi/32,
							R4, 31*np.pi/32,
							R4, 29*np.pi/32,
							R4, 25*np.pi/32,
							R4, 27*np.pi/32,
							R4, 17*np.pi/32,
							R4, 19*np.pi/32,
							R4, 23*np.pi/32,
							R4, 21*np.pi/32,
							R4, 63*np.pi/32,
							R4, 61*np.pi/32,
							R4, 57*np.pi/32,
							R4, 59*np.pi/32,
							R4, 49*np.pi/32,
							R4, 51*np.pi/32,
							R4, 55*np.pi/32,
							R4, 53*np.pi/32,
							R4, 33*np.pi/32,
							R4, 35*np.pi/32,
							R4, 39*np.pi/32,
							R4, 37*np.pi/32,
							R4, 47*np.pi/32,
							R4, 45*np.pi/32,
							R4, 41*np.pi/32,
							R4, 43*np.pi/32,
							R3, 1*np.pi/32,
							R3, 3*np.pi/32,
							R3, 7*np.pi/32,
							R3, 5*np.pi/32,
							R3, 15*np.pi/32,
							R3, 13*np.pi/32,
							R3, 9*np.pi/32,
							R3, 11*np.pi/32,
							R3, 31*np.pi/32,
							R3, 29*np.pi/32,
							R3, 25*np.pi/32,
							R3, 27*np.pi/32,
							R3, 17*np.pi/32,
							R3, 19*np.pi/32,
							R3, 23*np.pi/32,
							R3, 21*np.pi/32,
							R3, 63*np.pi/32,
							R3, 61*np.pi/32,
							R3, 57*np.pi/32,
							R3, 59*np.pi/32,
							R3, 49*np.pi/32,
							R3, 51*np.pi/32,
							R3, 55*np.pi/32,
							R3, 53*np.pi/32,
							R3, 33*np.pi/32,
							R3, 35*np.pi/32,
							R3, 39*np.pi/32,
							R3, 37*np.pi/32,
							R3, 47*np.pi/32,
							R3, 45*np.pi/32,
							R3, 41*np.pi/32,
							R3, 43*np.pi/32,
							R8, 1*np.pi/32,
							R8, 3*np.pi/32,
							R8, 7*np.pi/32,
							R8, 5*np.pi/32,
							R8, 15*np.pi/32,
							R8, 13*np.pi/32,
							R8, 9*np.pi/32,
							R8, 11*np.pi/32,
							R8, 31*np.pi/32,
							R8, 29*np.pi/32,
							R8, 25*np.pi/32,
							R8, 27*np.pi/32,
							R8, 17*np.pi/32,
							R8, 19*np.pi/32,
							R8, 23*np.pi/32,
							R8, 21*np.pi/32,
							R8, 63*np.pi/32,
							R8, 61*np.pi/32,
							R8, 57*np.pi/32,
							R8, 59*np.pi/32,
							R8, 49*np.pi/32,
							R8, 51*np.pi/32,
							R8, 55*np.pi/32,
							R8, 53*np.pi/32,
							R8, 33*np.pi/32,
							R8, 35*np.pi/32,
							R8, 39*np.pi/32,
							R8, 37*np.pi/32,
							R8, 47*np.pi/32,
							R8, 45*np.pi/32,
							R8, 41*np.pi/32,
							R8, 43*np.pi/32,
							R7, 1*np.pi/32,
							R7, 3*np.pi/32,
							R7, 7*np.pi/32,
							R7, 5*np.pi/32,
							R7, 15*np.pi/32,
							R7, 13*np.pi/32,
							R7, 9*np.pi/32,
							R7, 11*np.pi/32,
							R7, 31*np.pi/32,
							R7, 29*np.pi/32,
							R7, 25*np.pi/32,
							R7, 27*np.pi/32,
							R7, 17*np.pi/32,
							R7, 19*np.pi/32,
							R7, 23*np.pi/32,
							R7, 21*np.pi/32,
							R7, 63*np.pi/32,
							R7, 61*np.pi/32,
							R7, 57*np.pi/32,
							R7, 59*np.pi/32,
							R7, 49*np.pi/32,
							R7, 51*np.pi/32,
							R7, 55*np.pi/32,
							R7, 53*np.pi/32,
							R7, 33*np.pi/32,
							R7, 35*np.pi/32,
							R7, 39*np.pi/32,
							R7, 37*np.pi/32,
							R7, 47*np.pi/32,
							R7, 45*np.pi/32,
							R7, 41*np.pi/32,
							R7, 43*np.pi/32,
							R5, 1*np.pi/32,
							R5, 3*np.pi/32,
							R5, 7*np.pi/32,
							R5, 5*np.pi/32,
							R5, 15*np.pi/32,
							R5, 13*np.pi/32,
							R5, 9*np.pi/32,
							R5, 11*np.pi/32,
							R5, 31*np.pi/32,
							R5, 29*np.pi/32,
							R5, 25*np.pi/32,
							R5, 27*np.pi/32,
							R5, 17*np.pi/32,
							R5, 19*np.pi/32,
							R5, 23*np.pi/32,
							R5, 21*np.pi/32,
							R5, 63*np.pi/32,
							R5, 61*np.pi/32,
							R5, 57*np.pi/32,
							R5, 59*np.pi/32,
							R5, 49*np.pi/32,
							R5, 51*np.pi/32,
							R5, 55*np.pi/32,
							R5, 53*np.pi/32,
							R5, 33*np.pi/32,
							R5, 35*np.pi/32,
							R5, 39*np.pi/32,
							R5, 37*np.pi/32,
							R5, 47*np.pi/32,
							R5, 45*np.pi/32,
							R5, 41*np.pi/32,
							R5, 43*np.pi/32,
							R6, 1*np.pi/32,
							R6, 3*np.pi/32,
							R6, 7*np.pi/32,
							R6, 5*np.pi/32,
							R6, 15*np.pi/32,
							R6, 13*np.pi/32,
							R6, 9*np.pi/32,
							R6, 11*np.pi/32,
							R6, 31*np.pi/32,
							R6, 29*np.pi/32,
							R6, 25*np.pi/32,
							R6, 27*np.pi/32,
							R6, 17*np.pi/32,
							R6, 19*np.pi/32,
							R6, 23*np.pi/32,
							R6, 21*np.pi/32,
							R6, 63*np.pi/32,
							R6, 61*np.pi/32,
							R6, 57*np.pi/32,
							R6, 59*np.pi/32,
							R6, 49*np.pi/32,
							R6, 51*np.pi/32,
							R6, 55*np.pi/32,
							R6, 53*np.pi/32,
							R6, 33*np.pi/32,
							R6, 35*np.pi/32,
							R6, 39*np.pi/32,
							R6, 37*np.pi/32,
							R6, 47*np.pi/32,
							R6, 45*np.pi/32,
							R6, 41*np.pi/32,
							R6, 43*np.pi/32]

				# loop through all data
				for n in range(int(np.ceil(number_of_bits/bits_per_symbol))):
					
					# combine three bits and map to a complex symbol
					symbol_int = 0
					for i in range(bits_per_symbol):
						symbol_int += 2**i * data[bits_per_symbol*n + i]

					symbol = bit_map[symbol_int][0] * np.exp(1j*bit_map[symbol_int][1])

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

