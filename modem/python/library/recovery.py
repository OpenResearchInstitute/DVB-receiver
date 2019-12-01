import numpy as np
import fir_filter
import pi_filter
from scipy import signal

class recovery:
    """General purpose phase/frequency recovery object based on PLL principles """
    
      
    def __init__(   self, 
                    filter_bandwidth, 
                    filter_order, 
                    proportional_gain,
                    integral_gain):
        """ Create the general recovery object """
        
        # store the parameteers internally - important for stability analysis later
        self.filter_bandwidth = filter_bandwidth
        self.filter_order = filter_order
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain


        # calculate the FIR coefficients
        self.fir_coefficients = signal.firwin(  numtaps = self.filter_order,
                                                cutoff = self.filter_bandwidth)

        self.fir = fir_filter.fir_filter(   coeffs = self.fir_coefficients,
                                            complex = False)

        self.pi = pi_filter.pi_filter(  p_gain = self.proportional_gain,
                                        i_gain = self.integral_gain)



    def __write_coefficients(self, filename, wordlength, coefficients):
        """ Write a list of coefficients to a file with the correct wordlength
            scaling for a 2s complement digital number """

        # create the files
        f = open(filename, 'w')

        coefficients_max = max(coefficients)

        for coeff_float in coefficients:
    
            # convert first coefficient to twos complement binary
            if coeff_float < 0:
                coeff_fixed = format((2**wordlength) - abs(int(0.5*0.99999*coeff_float*2**wordlength/coefficients_max)), '016b')
            else:
                coeff_fixed = format(int(0.5*0.99999*coeff_float*2**wordlength/coefficients_max), '016b')

            # write the coeffients to file
            f.write(coeff_fixed+'\n')

        f.close()



    def write_coefficients_to_file( self,
                                    filename_prefix = "",
                                    fir_filename = "fir_coefficients.txt",
                                    pi_filename = "pi_coefficients.txt",
                                    fir_wordlength = 16,
                                    pi_wordlength = 16):
        """ Write the coefficients used by the recovery object to text files that
            can be read by other programs, ie. VHDL simulators """

        # write the FIR filter coefficients
        self.__write_coefficients(  filename = filename_prefix + fir_filename,
                                    wordlength = fir_wordlength,
                                    coefficients = self.fir_coefficients)

        # write the PI filter coefficients
        pi_coefficients = [self.proportional_gain, self.integral_gain]
        self.__write_coefficients(  filename = filename_prefix + pi_filename,
                                    wordlength = pi_wordlength,
                                    coefficients = pi_coefficients)