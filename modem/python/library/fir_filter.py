import numpy as np

class fir_filter:
    'FIR filter object'
    
      
    def __init__(self, coeffs, complex=False):
        
        # save the filter coeffs
        self.coeffs = list(reversed(coeffs))
        self.length = len(coeffs)
        self.complex = complex
        
        # create a buffer
        if self.complex:
            self.buffer = np.zeros(self.length) + 1j*np.zeros(self.length)
        else:
            self.buffer = np.zeros(self.length)


    def update_coeffs(self, coeffs, complex=False):
        
        # save the filter coeffs
        self.coeffs = coeffs
        self.length = len(coeffs)
        self.complex = complex
        
        # create a buffer
        if self.complex:
            self.buffer = np.zeros(self.length) + 1j*np.zeros(self.length)
        else:
            self.buffer = np.zeros(self.length)
     

    # input a new sample to update the filter state and output a new sample
    def update(self, input_sample):
        
        # update buffer
        for n in range(self.length-1):
            self.buffer[n] = self.buffer[n+1]
        self.buffer[self.length-1] = input_sample
        
        # compute dot product of buffer and filter coefficients
        return np.dot(self.coeffs, self.buffer)
       
    
    # reset the internal state
    def reset(self):
        if self.complex:
            self.buffer = np.zeros(self.length) + 1j*np.zeros(self.length)
        else:
            self.buffer = np.zeros(self.length)
        
    def print_buffer(self):
        for i in self.buffer:
            print(i)
        