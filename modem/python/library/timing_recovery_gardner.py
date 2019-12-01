import numpy as np
from recovery import recovery
import farrow_resampler

class timing_recovery_gardner(recovery):
    """Timing recovery object using the Garnder algorithm for timing error"""

    def __init__(   self, 
    				modulation_type,
                    filter_bandwidth, 
                    filter_order, 
                    proportional_gain,
                    integral_gain,
                    samples_per_symbol,
                    offset = False):
        """ Create the classical Costas loop carrier recovery object """

        recovery.__init__(  self, 
                            filter_bandwidth = filter_bandwidth, 
                            filter_order = filter_order, 
                            proportional_gain = proportional_gain,
                            integral_gain = integral_gain)

        # save the input parameters internally
        self.samples_per_symbol = samples_per_symbol
        self.modulation_type = modulation_type

        # select whether offset sampling is required
        if self.modulation_type in {"OQPSK", "GMSK"}:
        	self.offset = 1
        else:
        	self.offset = 0

        # initialise the phase and sample counter
        self.phi = 0.0
        self.count = 0
        self.sample_skip = 0
        self.i_n_1T = 0.0
        self.i_nT = 0.0
        self.i_n_T2 = 0.0
        self.q_n_1T = 0.0
        self.q_nT = 0.0
        self.q_n_T2 = 0.0
        self.error_filtered = 0.0

        # create the resampling object
        self.resampler = farrow_resampler.farrow_resampler(max_delay = 1, complex = True)


    def update(self, input_sample):
        """ process a new sample and create an estimated carrier correction """

        # resample the input signal
        resampled = self.resampler.update(input_sample, self.phi)

        output_tick = 0
    
        # update the sampling counter
        #
        #		if timing skip = -1 we need to retard the sampling by sample, keep the same count
        #		if timing skip = 1 we need to advance the sample by a sample, add two to the count
        #		otherwise operate normally and just adanvce the count by one
        #
        if self.sample_skip == -1:
            self.count = self.count
        elif self.sample_skip == 1:
            self.count = (self.count + 2) % self.samples_per_symbol
        else:
            self.count = (self.count + 1) % self.samples_per_symbol

        # output tick array, index 0 is the I channel and index 1 is the Q channel
        output_tick = [0, 0];

    
        # offset sampling point
        if self.count == 0.5 * self.samples_per_symbol:

            # sample the offset I stream
            self.i_n_T2 = np.real(resampled)

            # depending on if using and offset modulation calculate Q error or sample offset
            if self.offset:
                self.q_n_1T = self.q_nT
                self.q_nT = np.imag(resampled)
                self.q_error = (self.q_nT - self.q_n_1T) * self.q_n_T2
            else:
                self.q_n_T2 = np.imag(resampled)
                output_tick[1] = 1

            # flag this is a symbol transition
            output_tick[0] = 1
    

        # bit sampling point
        if self.count == 0 and self.sample_skip != -1:
        
            # calculate the timing error
            self.i_n_1T = self.i_nT
            self.i_nT = np.real(resampled)
            self.i_error = (self.i_nT - self.i_n_1T) * self.i_n_T2

            # depending on if using and offset modulation calculate Q error or sample offset
            if self.offset:
                self.q_n_T2 = np.imag(resampled)
                output_tick[1] = 1
            else:
                self.q_n_1T = self.q_nT
                self.q_nT = np.imag(resampled)
                self.q_error = (self.q_nT - self.q_n_1T) * self.q_n_T2

            # combine the I and Q errors
            error = self.i_error + self.q_error
        
            # filter the error signal and create a control signal
            self.error_filtered = self.fir.update(error)
            control_signal = self.pi.update(self.error_filtered)

            # update the new phase
            self.phi = self.phi + control_signal

            # if the phase is greater than one or less than zero we need to perform a sample skip
            if self.phi > 1:
                self.sample_skip = 1
                self.phi = self.phi - 1
            elif self.phi < 0:
                self.sample_skip = -1
                self.phi = self.phi + 1
            else:
                self.sample_skip = 0

        else:
        	self.sample_skip = 0


        # return both the retimed data and the error
        return resampled, output_tick, self.error_filtered