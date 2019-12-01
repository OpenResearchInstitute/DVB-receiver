import numpy as np
import copy

class farrow_resampler:
    'Farrow Lagrange interpolation resampler object'
    
      
    def __init__(self, max_delay, complex=False):
        
        # save the interpolators parameters
        self.max_delay = max_delay
        self.complex = complex
        
        # create the delay line and previous sample  
        if self.complex:
            # self.integer_delay_shift = np.zeros(max_delay+1) + 1j*np.zeros(max_delay+1)
            # self.delay_line = np.zeros(4) + 1j*np.zeros(4)

            self.integer_delay_shift = np.zeros(max_delay+1, dtype=np.complex)
            self.delay_line = np.zeros(4, dtype=np.complex)
        else:
            self.integer_delay_shift = np.zeros(max_delay)
            self.delay_line = np.zeros(4)

        
        
    # input a new sample to update the filter state and output a new sample
    def update(self, input_sample, delay):
        
        # split the delay into integer and fractional parts
        integer_delay = int(delay/1)
        fractional_delay = delay%1

        # print(integer_delay)
            
        # update the integer delay line
        if integer_delay >= 1:
            for i in range(0,integer_delay):
                self.integer_delay_shift[integer_delay-i] = self.integer_delay_shift[integer_delay-i-1]
            self.integer_delay_shift[0] = input_sample
        else:
            self.integer_delay_shift[integer_delay] = input_sample


        # update the Farrow delay line
        self.delay_line[3] = self.delay_line[2]
        self.delay_line[2] = self.delay_line[1]
        self.delay_line[1] = self.delay_line[0]
        self.delay_line[0] = self.integer_delay_shift[integer_delay]

        # calculate intermediate values
        b0 = (self.delay_line[0] - self.delay_line[3])/6 + (self.delay_line[2] - self.delay_line[1])/2
        b1 = self.delay_line[0] - self.delay_line[1] - b0
        b2 = (self.delay_line[0] - self.delay_line[2])/2
        a0 = self.delay_line[1]
        a1 = b2 - b0
        a2 = b1 - a1
        a3 = b0

        # calculate the output
        sample = a0 + -fractional_delay*(a1 + -fractional_delay*(a3*-fractional_delay + a2) )
       
        return sample
       
    
    # reset the internal state
    def reset(self):
        # create the delay line     
        if self.complex:
            self.integer_delay_shift = np.zeros(max_delay) + 1j*np.zeros(max_delay)
            self.delay_line = np.zeros(4) + 1j*np.zeros(4)
        else:
            self.integer_delay_shift = np.zeros(max_delay)
            self.delay_line = np.zeros(4)
        
        # store the previous sample
        self.integer_delay_previous = 0.0
