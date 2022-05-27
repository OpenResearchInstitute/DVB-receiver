"""
MIT License

Copyright (c) 2020 Koliber Engineering, koliber.eng@gmail.com
Copyright (c) 2020 Koliber Engineering, koliber.eng@gmail.com
Copyright (c) 2022 Koliber Engineering, koliber.eng@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import random as rand
from myrcf import myrcf

def awgn(input_signal, snr_dB, rate=1.0):
    """
    Addditive White Gaussian Noise (AWGN) Channel.

    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.

    snr_dB : float
        Output SNR required in dB.

    rate : float
        Rate of the a FEC code used if any, otherwise 1.

    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(abs(input_signal) * abs(input_signal))/len(input_signal)
    snr_linear = 10**(snr_dB/10.0)
    noise_variance = avg_energy/(2*rate*snr_linear)

    if input_signal.dtype == complex:
        noise = (np.sqrt(noise_variance) * nprand.randn(len(input_signal))) + (np.sqrt(noise_variance) * nprand.randn(len(input_signal))*1j)
    else:
        noise = np.sqrt(2*noise_variance) * nprand.randn(len(input_signal))

    output_signal = input_signal + noise

    return output_signal

def add_frequency_offset(waveform, Fs, delta_f, phaseoff=0):
    """
    Add frequency offset impairment to input signal.

    Parameters
    ----------
    waveform : 1D ndarray of floats
        Input signal.

    Fs : float
        Sampling frequency (in Hz).

    delta_f : float
        Frequency offset (in Hz).

    phaseoff : float
        phase offset in radians

    Returns
    -------
    output_waveform : 1D ndarray of floats
        Output signal with frequency offset.
    """

    output_waveform = waveform*np.exp(1j*2*np.pi*(delta_f/Fs)*np.arange(len(waveform)) + \
                                      1j * phaseoff)
    return output_waveform


def bin2BPSK(binData):
    # map the binary dat to inphase values (real)
    for i in range(len(binData)):
        if (binData[i] == 1):
            bpskData = 1.0
        else:
            bpskData = -1.0

    bpskData = np.array(bpskData, dtype=complex)
    return bpskData


def myrcf(Fd, OverSamp, TYPE, alpha, N_symb):
    ## %%%%%%%%%>>>> for debug %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # import matplotlib.pyplot as plt
    # Fd = 1;
    # OverSamp = 8;
    # TYPE = 'normal';
    # alpha = 0.5;
    # N_symb = 6 ;
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #hhx = myrcf(1, 8, 'sqrt', 0.5, 6 )

    # % Fd not used.
    # %TYPE not used.

    if (Fd == 1):
        OvrSampRate = OverSamp
        #%N_symb                            #% number of symbols wide from center
        N_samples = 2*N_symb*OvrSampRate    #% samples
        p0 = 0
        #if strfind(TYPE, 'normal')
        if (TYPE == 'normal'):
            #% defining the sinc filter
            sincNum = np.sin(np.pi*(np.arange(-N_samples/2, (N_samples/2)+1, 1))/OvrSampRate)   #% numerator of the sinc function
            sincDen = (np.pi*(np.arange(-N_samples/2, (N_samples/2)+1, 1))/OvrSampRate)     #% denominator of the sinc function
            sincDenZero = N_samples/2     #was: sincDenZero = find(abs(sincDen) < 10^-10)
            sincOp = sincNum/sincDen      #was: sincOp = sincNum./sincDen
            sincOp[sincDenZero] = 1         #% sin(pix/(pix) =1 for x =0
            
            
            #was: cosNum = cos(alpha*np.pi*[-N_samples/2:1:N_samples/2]/OvrSampRate)        
            cosNum = np.cos(alpha*np.pi*(np.arange(-N_samples/2, (N_samples/2)+1, 1))/OvrSampRate)
            #was: cosDen = (1-(2*alpha*[-N_samples/2:1:N_samples/2]/OvrSampRate).^2)
            cosDen = (1-(2*alpha*(np.arange(-N_samples/2, (N_samples/2)+1,1))/OvrSampRate)**2)
            
            #cosDenZero = find(abs(cosDen)<10**-10)
            cosDenZero = np.where(abs(cosDen) < 10**-10)
            cosOp = cosNum/cosDen
            cosOp[cosDenZero] = np.pi/4
            gt_alpha = sincOp*cosOp
            HH = gt_alpha 
            #%GF_alpha = fft(gt_alpha,1024)
    
        #elseif strfind(TYPE ,'sqrt')
        elif (TYPE == 'sqrt'):
            #%Limiting the response to +/- N_symb (default to -6T to 6T)
            #was: t = -N_symb : 1/OvrSampRate : N_symb; 
            t = np.linspace(-N_symb, N_symb, num=N_symb*2*OvrSampRate+1, endpoint=True)
            
            #%This can be increased or decreased according to the requirement
            pp=np.zeros((np.size(t)), dtype=float)
            
            for i in range(np.size(t)): 
                if (t[i] == 0) :
                        pp[i]= (1-alpha +(4*alpha/np.pi))
                        p0 = pp[i]
                elif ((t[i] == 1/(4*alpha)) | (t[i] == -1/(4*alpha))):
                        pp[i] =     alpha/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*alpha))+ \
                        (1-(2/np.pi))*np.cos(np.pi/(4*alpha)))
                else :
                        pp[i] =     (np.sin(np.pi*t[i]*(1-alpha))+ \
                                4*alpha*t[i]*np.cos(np.pi*t[i]*(1+alpha)))/ \
                                (np.pi*t[i]*(1-(4*alpha*t[i])**2))
    
            HH = pp/p0          #% Normalize to one.
            # response=p./sqrt(sum(p.^2)); %Normalization to unit energy <- matlab code
    
        return HH

