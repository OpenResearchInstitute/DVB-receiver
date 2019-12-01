import numpy as np
import iir_filter
import pi_filter

class demodulator:
    'General purpose demodulator that supports BPSK, QPSK and OQPSK'
    
      
    def __init__(self, modulation_type, samples_per_symbol):
        """ Create the classical Costas loop carrier recovery object """
        
        # store the parameteers internally - important for stability analysis later
        self.modulation_type = modulation_type
        self.samples_per_symbol = samples_per_symbol

        # create the sample counter
        self.count = 0

        # I and Q channel sum variables
        self.I_sum = 0.0
        self.Q_sum = 0.0



    def update(self, input_sample, input_tick):
        """ process a new sample, estimate a new demodulated bit if the correct time """

        # # if the previous block wants to delay sampling it will supply an empty list
        # #  therefore we want to skip any operation and hold back on advancing the count
        # if input_sample != []:

        #     # new bit transition, return demodulated bits depending on modulation type
        #     if self.count == 0:

        #         self.count += 1

        #         if self.modulation_type == "BPSK":
        #             return [np.real(input_sample)]

        #         elif self.modulation_type == "QPSK":
        #             return [np.real(input_sample), np.imag(input_sample)]

        #         elif self.modulation_type == "OQPSK":
        #             return [np.real(input_sample)]


        #     # offset bit, return demodulated bit for the offset bit in OQPSK
        #     elif self.count == self.samples_per_symbol/2:

        #         self.count += 1

        #         if self.modulation_type == "OQPSK":
        #             return [np.imag(input_sample)]

        #     # not the correct time demodulate, return nothing
        #     # callign function should be used with the extend function rather than append so a zero length list is added
        #     else:
        #         self.count += 1
        #         return []

        # else:
        #     return []


        if output_tick[0] == 1:
            I_sample = I_sum
            I_sum = 0.0
            



