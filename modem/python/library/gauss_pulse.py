import math
import numpy as np
import scipy as sp
from scipy import special
from copy import deepcopy


"""
    Functions to provide Gausssuan type filter coefficients to be used
    in filters;.
"""

def gauss_pulse(sps = 32, BT = 0.5):
    """ Generate a standard Gaussian pulse with the specified parameters """

    g_pulse = []
    ntaps = 4*sps
    scale = 0
    dt = 1.0/sps
    s = 1.0/(math.sqrt(np.log(2.0)) / (2*math.pi*BT));
    t0 = -0.5 * ntaps;

    for i in range(ntaps):
        t0 += 1
        ts = s*dt*t0
        g_pulse.append(np.exp(-0.5*ts*ts))
        scale += g_pulse[i]
    
    g_pulse = [_ / scale for _ in g_pulse]

    return g_pulse



def gauss_laurent_amp(sps = 32, BT = 0.5):
    """ Generate the Laurent AMP (Amplitude Modulation Pulse) decompostion for the Gaussian pulse """

    # modulation parameters
    T = 1.0
    h = 0.5
    B = BT / T
    L = 3

    # generate sampling points for L symbol periods
    t_len = 0
    t_start = -L*T/2 + T/(2*sps)
    t_stop = L*T/2

    # generate Gaussian pulse
    g = np.zeros(int(L*sps))
    g_sum = 0

    t_incr = T/sps
    ti = t_start


    # calculate the Gaussian pulse
    g[0] = 0

    while ti < t_stop:

        g[t_len] =  0.5*special.erfc((2.0*np.pi*B*(ti - T/2.0)/(np.sqrt(np.log(2.0))))/np.sqrt(2.0)) - 0.5*special.erfc((2.0*np.pi*B*(ti + T/2.0)/(np.sqrt(np.log(2.0))))/np.sqrt(2.0))

        g_sum = g_sum + g[t_len]
        ti += t_incr
        t_len += 1


    # normalise the Gaussian pulses amplitude
    for i in range(t_len):
        g[i] = g[i] / g_sum * np.pi/2
    

    # integrate phase
    q = np.zeros(int(t_len))

    q[0] = 0
    q[1] = g[0]

    for i in range (2, t_len):
        q[i] = q[i-1] + g[i-1]
    

    # compute two sided generalized phase pulse function
    s = np.zeros(int(2*(t_len+1)))

    for i in range(t_len):
        s[i] = np.sin(q[i]) / np.sin(np.pi*h)
    
    for i in range(t_len, 2*t_len):
        s[i] = np.sin(np.pi*h - q[int(i - (t_len+1)+1)]) / np.sin(np.pi*h)


    # compute C0 pulse: valid for all L values
    c0_len = 2*(t_len+1)-1 - sps*(L-1)

    taps = deepcopy(s[:c0_len])

    # calculate the pulse
    for i in range(1, L):
        for j in range(c0_len):
            taps[j] = taps[j] * deepcopy(s[i*sps + j])

    # pass back the number of coefficients in the filter
    return taps