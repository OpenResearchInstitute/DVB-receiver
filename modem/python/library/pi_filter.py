import numpy as np

class pi_filter:
    'Proportional integral filter object'
    
      
    def __init__(self, p_gain, i_gain, complex=False):
        
        # save the filter parameters
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.complex = complex
        
        # create an integral summing variable
        if self.complex:
            self.i_sum = 0.0 + 1j*0.0
        else:
            self.i_sum = 0.0
        
    # input a new sample to update the filter state and output a new sample
    def update(self, input_sample):
        
        # calculate the outpur
        self.i_sum = self.i_sum + input_sample
        return input_sample * self.p_gain + self.i_sum * self.i_gain

       
    
    # reset the internal state
    def reset(self):
        # create an integral summing variable
        if self.complex:
            self.i_sum = 0.0 + 1j*0.0
        else:
            self.i_sum = 0.0
        
    def print_buffer(self):
        print("Pgain = ", self.p_gain, "   Igain = ", self.i_gain, "   Isum = ", self.i_sum)
        