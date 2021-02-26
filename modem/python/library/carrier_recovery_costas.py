import numpy as np
from recovery import recovery
import control
import matplotlib.pyplot as plt

class carrier_recovery_costas(recovery):
    """ A costas loop implementation for carrier phase/frequency recovery """

    def __init__(   self,
                    modulation_type,
                    filter_bandwidth, 
                    filter_order, 
                    proportional_gain,
                    integral_gain):
        """ Create the classical Costas loop carrier recovery object """

        recovery.__init__(	self, 
                            filter_bandwidth = filter_bandwidth, 
                            filter_order = filter_order, 
                            proportional_gain = proportional_gain,
                            integral_gain = integral_gain)

        # store the modulation type
        self.modulation_type = modulation_type

        # initialise the phase
        self.phi = 0.0



    def update(self, input_sample):
        """ Update the object by processing a new sample and create an 
            estimated carrier correction """
        
        # mix with LO
        lo = np.exp(-1j*(self.phi))
        mixed = lo * input_sample

        # split the mixed signal into real and imaginary components
        real = np.real(mixed)
        imag = np.imag(mixed)
        real_sgn = np.sign(real)
        imag_sgn = np.sign(imag)

        # ensure that the sign is either -1 or 1
        if real_sgn == 0.0:
            real_sgn = 1.0
        if imag_sgn == 0.0:
            imag_sgn = 1.0

        # select the correct error function
        if self.modulation_type == "BPSK":
            self.error = real * imag
        elif self.modulation_type == "QPSK":
            i_arm = real * imag_sgn
            q_arm = imag * real_sgn
            self.error = i_arm - q_arm
        elif self.modulation_type in {"OQPSK", "GMSK"}:
            i_arm = real * imag_sgn
            q_arm = imag * real_sgn
            self.error = i_arm - q_arm

        # filter the error
        # error_filtered = self.iir.update(error)
        self.error_filtered = self.fir.update(self.error)
        self.control_signal = self.pi.update(self.error_filtered)
    
        # create the new LO outputs
        self.phi  -= self.control_signal

        return mixed

        
    def stability_analysis(self):
        """ Perform a stability analysis of the loop with 
            settings defined in the constructor """

        # create the phase detector transfer function
        K_pd = control.tf([1], [1])

        # create the low pass filter transfer function
        iir_taps = self.iir.read_coeffs()
        iir_b = iir_taps[0:2]
        iir_a = iir_taps[3:5]
        K_lpf = control.tf(iir_b, iir_a)

        # create the PI filter transfer function
        K_pi = control.tf([self.proportional_gain, self.integral_gain],[1, 0])

        # create the VCO transfer function
        K_vco = control.tf([1], [1, 0])

        # create the final transfer function
        K_ol = K_pd * K_lpf * K_pi * K_vco

        # find the bode plot
        mag, phase, omega = control.bode(K_ol)

        # plot the bode plot
        plt.plot(omega/(2*np.pi),mag)
        plt.plot(omega/(2*np.pi),phase)
        plt.show()

        # find the gain and phase margins
        gm, pm, wg, wp = control.margin(K_ol)
        print("Gain margin = ", gm, "Phase margin = ", pm)

        print(K_ol)