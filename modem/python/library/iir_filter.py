import numpy as np

class iir_filter:
    'SOS IIR filter object'
    
    def __init__(self, bandwidth, damping, gain, complex=False):
        """ Create the IIR filter by specifying the second order
            control system paremeters """
        
        # save the user filter parameters
        self.bandwidth = bandwidth
        self.damping = damping
        self.gain = gain
        
        # generate loop filter parameters (active PI design)        
        C = 1.0/(np.tan(np.pi*self.bandwidth))
        self.b0 = 1.0/(1.0 + 2.0*self.damping*C + C*C)
        self.b1 = 2.0*self.b0
        self.b2 = self.b0
        self.a0 = 1.0
        self.a1 = 2.0*self.b0*(1-C*C)
        self.a2 = self.b0*(1 - 2*self.damping*C + C*C)
        
        # filter buffer
        if complex:
            self.v0 = 0.0 + 1j*0.0
            self.v1 = 0.0 + 1j*0.0
            self.v2 = 0.0 + 1j*0.0
        else:
            self.v0 = 0.0
            self.v1 = 0.0
            self.v2 = 0.0
            
        
    def update(self, input_sample):
        """ input a new sample to update the filter state and output a new sample """
        self.v2 = self.v1
        self.v1 = self.v0
        self.v0 = input_sample - self.v1*self.a1 - self.v2*self.a2
        
        return self.v0*self.b0 + self.v1*self.b1 + self.v2*self.b2

    
    def reset(self):
        """ reset the internal state """
        self.v0 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0
        

    def set_coeffs(self, a, b):
        """ set the coefficients """
        
        # set the poles
        self.a1 = a[1]
        self.a2 = a[2]
        
        # set the zeros
        self.b0 = b[0]
        self.b1 = b[1]
        self.b2 = b[2]
        
    def read_coeffs(self):
        """ read back the coefficients """
        return [self.b0, self.b1, self.b2, 1.0, self.a1, self.a2]