"""
modem_2101_timing sim 1
timing recovery PLL sim 
This sim walks through all the stages of the polyphase matched filter
shows time series, constellation and eye-diagrams for each path option
then a timing loop determines which path is the correct path

just below "if __name__ == '__main__':" line

can add a phase offset                   
can also add a small frequency offset     
can also enable SNR amount of noise

Notes: currently this has bugs in the phase locked loop parameters. 
"""
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
import pandas as pd
from scipy import signal as sig
# possibly use matplotlib instead of plotly
# import matplotlib.pyplot as plt  # not using matplotlib
# using plotly instead
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo

# import time
import visualization
from comms_utils import add_frequency_offset, awgn, myrcf


class ModulateDemodulate:
    def __init__(self):
        self.over_sample = 8
        self.alpha = 0.5
        self.modulation = 'qpsk'
        self.code_rate = None
        self.phase_offset = 0.05 * 2 * np.pi
        self.timing_idx_start = 5
        self.accum_t_sv = None

        # qpsk
        self.r_ang_qpsk = {
            0: [0,       np.pi /  4.0],  # 0
            1: [0, 7.0 * np.pi /  4.0],  # 1
            2: [0, 3.0 * np.pi /  4.0],  # 2
            3: [0, 5.0 * np.pi /  4.0]}  # 3
        # 8 psk 
        self.r_ang_8psk = {
            0: [0,       np.pi /  4.0], # 0   4 
            1: [0,                0.0], # 1   5 
            2: [0, 4.0 * np.pi /  4.0], # 2   6 
            3: [0, 5.0 * np.pi /  4.0], # 3   7 
            4: [0, 2.0 * np.pi /  4.0], # 4   8 
            5: [0, 7.0 * np.pi /  4.0], # 5   9 
            6: [0, 3.0 * np.pi /  4.0], # 6   10
            7: [0, 6.0 * np.pi /  4.0]} # 7   11

        # 16APSK (angles only)
        self.r_ang_16apsk = {
            # r1 , outer ring
            0 : [1,         np.pi /  4.0], # 0   12
            1 : [1,        -np.pi /  4.0], # 1   13
            2 : [1,   3.0 * np.pi /  4.0], # 2   14
            3 : [1,  -3.0 * np.pi /  4.0], # 3   15
            4 : [1,         np.pi / 12.0], # 4   16
            5 : [1,        -np.pi / 12.0], # 5   17
            6 : [1,  11.0 * np.pi / 12.0], # 6   18
            7 : [1, -11.0 * np.pi / 12.0], # 7   19
            8 : [1,   5.0 * np.pi / 12.0], # 8   20
            9 : [1,  -5.0 * np.pi / 12.0], # 9   21
            10: [1,   7.0 * np.pi / 12.0], # 10  22
            11: [1,  -7.0 * np.pi / 12.0], # 11  23
            # r0, inner ring
            12: [0,         np.pi /  4.0], # 12  24
            13: [0,        -np.pi /  4.0], # 13  25
            14: [0,   3.0 * np.pi /  4.0], # 14  26
            15: [0,  -3.0 * np.pi /  4.0]} # 15  27

        # 32APSK (angles only) r2 is 1.0
        self.r_ang_32apsk = {
            0 : [0,        np.pi /  4.0],  # 0    28
            1 : [0,  5.0 * np.pi / 12.0],  # 1    29
            2 : [0,       -np.pi /  4.0],  # 2    30
            3 : [0, -5.0 * np.pi / 12.0],  # 3    31
            4 : [0,  3.0 * np.pi /  4.0],  # 4    32
            5 : [0,  7.0 * np.pi / 12.0],  # 5    33
            6 : [0, -3.0 * np.pi /  4.0],  # 6    34
            7 : [0, -7.0 * np.pi / 12.0],  # 7    35
            8 : [2,        np.pi /  8.0],  # 8    36
            9 : [2,  3.0 * np.pi /  8.0],  # 9    37
            10: [2,       -np.pi /  4.0],  # 10   38
            11: [2,       -np.pi /  2.0],  # 11   39
            12: [2,  3.0 * np.pi /  4.0],  # 12   40
            13: [2,        np.pi /  2.0],  # 13   41
            14: [2, -7.0 * np.pi /  8.0],  # 14   42
            15: [2, -5.0 * np.pi /  8.0],  # 15   43
            16: [0,        np.pi / 12.0],  # 16   44
            17: [1,        np.pi /  4.0],  # 17   45
            18: [0,       -np.pi / 12.0],  # 18   46
            19: [1,       -np.pi /  4.0],  # 19   47
            20: [0, 11.0 * np.pi / 12.0],  # 20   48
            21: [1,  3.0 * np.pi /  4.0],  # 21   49
            22: [0,-11.0 * np.pi / 12.0],  # 22   50
            23: [1, -3.0 * np.pi /  4.0],  # 23   51
            24: [2,                 0.0],  # 24   52
            25: [2,        np.pi /  4.0],  # 25   53
            26: [2,       -np.pi /  8.0],  # 26   54
            27: [2, -3.0 * np.pi /  8.0],  # 27   55
            28: [2,  7.0 * np.pi /  8.0],  # 28   56
            29: [2,  5.0 * np.pi /  8.0],  # 29   57
            30: [2,        np.pi       ],  # 30   58
            31: [2, -3.0 * np.pi /  4.0]}  # 31   59
             
        # note the radius ratio in this implementation starts at radius 0
        # not radius 1, most documentation refer to it as R2/R1 or R3/R1
        self.r_ratio_16apsk = {  # R1/R0, R2/R0 (no R2)
            'c2/3': [3.15],
            'c3/4': [2.85],
            'c4/5': [2.75],
            'c5/6': [2.70],
            'c8/9': [2.60],
            'c9/10':[2.57]}

        self.r_ratio_32apsk = {  # R1/R0, R2/R0
            'c3/4' : [2.84, 5.27],
            'c4/5' : [2.72, 4.87],
            'c5/6' : [2.64, 4.64],
            'c8/9' : [2.54, 4.33],
            'c9/10': [2.53, 4.30]}
            
        self.n_const_symbols = {
            'qpsk': 4,
            '8psk': 8,
            '16apsk':16,
            '32apsk':32}

    def get_mag_list(self):
        n = self.n_const_symbols[self.modulation]
        mag_list = [None] * n
        r = [1] * 3
        r_ang_list = [None] * n
        if self.modulation == 'qpsk':
            r_ang_list = self.r_ang_qpsk
            r = [1]
        if self.modulation == '8psk':
            r_ang_list = self.r_ang_8psk
            r = [1]
        if self.modulation == '16apsk':
            r_ang_list = self.r_ang_16apsk
            r_ratio = self.r_ratio_16apsk[self.code_rate]
            r[0] = 1/r_ratio[0]
        if self.modulation == '32apsk':
            r_ang_list = self.r_ang_32apsk
            r_ratio = self.r_ratio_32apsk[self.code_rate]
            r[0] = 1/r_ratio[1]
            r[1] = r_ratio[0]/r_ratio[1]
        
        for idx in range(n):
            mag_list[idx] = r[r_ang_list[idx][0]]

        return mag_list

    def get_constellation(self):
        # returns an indexed list of IQ values based on symbol index
        n = self.n_const_symbols[self.modulation]
        mag_list = self.get_mag_list()
        
        if self.modulation == 'qpsk':
            ang_list = self.r_ang_qpsk
            # mag_list = self.mag_list_qpsk
        if self.modulation == '8psk':
            ang_list = self.r_ang_8psk
            # mag_list = self.mag_list_8psk
        if self.modulation == '16apsk':
            ang_list = self.r_ang_16apsk
            # mag_list = self.mag_list_16apsk
        if self.modulation == '32apsk':
            ang_list = self.r_ang_32apsk
            # mag_list = self.mag_list_32apsk
            
        iq_list = [None]*n
        for idx in range(n):
            real = mag_list[idx] * np.cos(ang_list[idx][1])
            imag = mag_list[idx] * np.sin(ang_list[idx][1])
            iq_list[idx] = [real, imag]
        [print(i, iq_list[i]) for i in range(len(iq_list))]
        return iq_list

    def polyphase_filter(self, over_sample, alpha, n_symbol):
        """
        creates a root raised cosine polyphase shaping filter .
        defaults to 8 path and 8 samples per symbol
        """
        # - 8-path Shaping Filter, 8-samples/symbols
        # matlab  hhx=rcosine(1,OverSamp,'sqrt',alpha,n_symb);
        hhx = myrcf(1, over_sample, 'sqrt', alpha, n_symbol )

        # zero extend for length to a multiple of over_sample
        hh = np.append(hhx, np.zeros(over_sample-1))

        # normalize the filter to the maximum value of the filter weights
        hh_max = np.amax(hh)
        hh = hh / hh_max

        # reshape filter to use as a polyphase filter
        width = (n_symbol * 2) + 1
        hh2 = np.reshape(hh, (over_sample, width), order='F')  # polyphase filter
        
        if DEBUGDERIVATIVEFILTER:
            vis = visualization.Visualization()
            vis.plot_data([hh],['hh'],points=True)
            
            nplots = hh2.shape[0]  # npaths
            plots = [None] * nplots
            for i in range(nplots):
                plots[i] = [0]*len(hh)
                for j in range(hh2.shape[1]):
                    plots[i][i+j*hh2.shape[0]] = hh2[i,j]
            plots.append(hh)

            vis.plot_data_1figure(plots,['hh2_'+ str(x) for x in range(nplots+1)],points=False)
            vis.plot_data_1figure([plots[0], hh],['plots 0', 'hh'],points=False)
            vis.plot_data(plots,['hh2_'+ str(x) for x in range(nplots)],points=False)

        return hh, hh2

    def polyphase_derivative_filter(self, n_paths, over_sample, alpha, n_symbol):
        # --- 32-path Polyphase Matched filter, 2-samples per symbol ------------
        # %hh_t=rcosine(1,64,'sqrt',0.5,6);                % Matched filter
        hh = myrcf(1,over_sample, 'sqrt', alpha, n_symbol)
        
        # normally leave this as true unless debugging the window
        APPLY_WINDOW = True
        if APPLY_WINDOW:
            window = sig.windows.blackmanharris(len(hh))
            hhw = hh * window
        else:
            hhw = hh

        x = n_paths - (len(hhw) % n_paths)
        hhw = np.append(hhw, np.zeros(x))    # zeros extending
        # hhw[0:len(hhw):]
        # scratch
        hhx = sig.firwin(len(hh), 13 / len(hh), window='blackmanharris')
        # end scratch

        # matlab code: Not sure how this scales for unity. hh is from the other filter
        # scl=hh_t[1::32]*hh(1:4:len(hh))'; #% scale for unity gain
        # hh_t=hh_t/scl;

        # normalize the filter to the maximum value of the filter weights
        hhw_max = np.amax(hh)
        scl = (hhw @ hhw.T) / n_paths
        hhwn = hhw / scl  # scale for unity gain???

        # note n_paths effectively adds gain to the derivative function, scales +-1 to gain value.
        dhh = np.convolve(hhwn, np.array([1, 0, -1])*n_paths)  # derivative matched filter
        #
        # hh2_length = (len(hhwn) + (n_paths - (len(hhwn)%n_paths)))/n_paths
        hh2_length = (len(hhwn) /n_paths)

        # matched filter and derivative matched filter outputs.
        # note: order='F' is to use the same order as matlab. reshape(row, column)
        # where n_paths are rows
        MF = hhwn.reshape((int(n_paths), int(hh2_length)), order='F') # 32 path polyphase MF
        dMF = dhh[2:].reshape((int(n_paths),int(hh2_length)), order='F')     # 32 path polyphase dMF

        if DEBUGDERIVATIVEFILTER:
            vis = visualization.Visualization()
            vis.plot_data_1figure([hhwn],['hhwn'],points=True)

            nplots = MF.shape[0]  # npaths
            plots = [None] * nplots
            for i in range(nplots):
                plots[i] = [0]*len(hhwn)
                for j in range(MF.shape[1]):
                    plots[i][i+j*MF.shape[0]] = MF[i,j]
            plots.append(hhwn)

            vis.plot_data_1figure(plots,['MF_'+ str(x) for x in range(nplots+1)],points=False)
            vis.plot_data_1figure([plots[0], hhwn],['plots 0', 'hhwn'],points=False)
            vis.plot_data_1figure(plots[:-1],['MF_'+ str(x) for x in range(nplots)],points=False)
            # -----------------------------------------------------------
            vis.plot_data([dhh],['dhh'],points=True)
            nplots = dMF.shape[0]  # npaths
            plots = [None] * nplots
            for i in range(nplots):
                plots[i] = [0]*len(dhh[2:])
                for j in range(dMF.shape[1]):
                    plots[i][i+j*dMF.shape[0]] = dMF[i,j]
            plots.append(dhh[2:])

            vis.plot_data_1figure(plots,['dMF_'+ str(x) for x in range(nplots+1)],points=False)
            vis.plot_data_1figure([plots[0], dhh[2:]],['plots 0', 'dhh[2:]'],points=False)
            vis.plot_data_1figure(plots[:-1],['dMF_'+ str(x) for x in range(nplots)],points=False)
            # -----------------------------------------------------------

            # fig = go.Figure().set_subplots(2,2)
            # fig = make_subplots(rows=2, cols=2)
            # fig.add_trace(go.Scatter(y=hhwn, name="hhwn"), row=1, col=1)
            # fig.add_trace(go.Scatter(y=dhh, name="dhh"), row=1, col=2)

            # nplots = MF.shape[0]  # npaths
            # for i in range(nplots):
            #     fig.add_trace(go.Scatter(y=MF[i, :],mode="lines+markers"), row=2, col=1)
            #     fig.add_trace(go.Scatter(y=dMF[i, :],mode="lines+markers"), row=2, col=2)
            # fig.update_layout(title="filters hhwn, dhh, MF, dMF")
            # pyo.plot(fig, filename="filters_hhwn_dhh_MF_dMF_plot.html")

            # ncol = 5
            # nrow = int(np.ceil(nplots / ncol))
            # fig = make_subplots(rows=nrow, cols=ncol)
            # for i in range(nplots):
            #     fig.add_trace(go.Scatter(y=MF[i,:]), row=int(np.floor(i/ncol))+1, col=(int(i % ncol)+1))
            # fig.update_layout(title="polyphase MF")
            # pyo.plot(fig, filename="polyphase_matched_filter_plot.html")
        return MF, dMF

    def generate_data(self, n_data, iq_list):
        # Input constellation
        # if self.modulation=='qpsk':
        #     x0=(np.floor(2*np.random.uniform(0, 1, n_data))-0.5)/0.5 + \
        #         1j*(np.floor(2*np.random.uniform(0,1,n_data))-0.5)/0.5
        #     return x0
        if self.modulation == 'qam':
            x0=(np.floor(4*np.random.uniform(0, 1, n_data))-1.5)/1.5 + \
                1j*(np.floor(4*np.random.uniform(0, 1, n_data))-1.5)/1.5
            return x0

        if self.modulation == 'qpsk':
            x0 = np.random.randint(0,4,size=n_data)

        if self.modulation == '8psk':
            x0 = np.random.randint(0,8,size=n_data)

        if self.modulation == '16apsk':
            x0 = np.random.randint(0,16,size=n_data)

        if self.modulation == '32apsk':
            x0 = np.random.randint(0,32,size=n_data)

        x1 = np.zeros(n_data,dtype=complex)
        for idx in range(len(x0)):
            x1[idx] = iq_list[x0[idx]][0] + 1j*iq_list[x0[idx]][1]
        return x1

        # print('Error: modulation is incorrect or not supported')
        # return None

    def shape_upsample(self, hh2, data):
        """
        Shape and up-sample input constellation
        hh2 filter is the polyphase filter should have dimension equal to
        filter paths x filter width
        data is an array of complex values.
        fpaths is an integer equal to the upsample value
        """
        fpaths = hh2.shape[0]  # number of filter paths
        flength = hh2.shape[1]     # each path filter length
        arraysize = len(data) * fpaths

        # preallocate memory to make things faster.
        reg=np.zeros((flength),dtype=complex)
        x1  = np.zeros((arraysize),dtype=complex)
        x1i = np.zeros((arraysize),dtype=float)
        x1q = np.zeros((arraysize),dtype=float)

        mm=0  # output clock index
        for nn in range(len(data)):
            # shift register new values in on the left , old values go out on the right
            reg = np.append(data[nn], reg[0:-1])
            for kk in range(fpaths):           # kk=1:8
                # matrix multiply the shift register by each
                # slice of the phase of the filter. each time
                # adding a point and therefore interpolating
                x1[mm+kk] = reg.dot(hh2[kk, :].conj())
                # used for debug only 
                # x1i(mm+kk)=real(reg)*hh2(kk,:)';
                # x1q(mm+kk)=imag(reg)*hh2(kk,:)';
            mm = mm + fpaths
        if DEBUGUPSAMPLE:
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(y=np.real(data), name="data real", mode="lines+markers"), row=1, col=1)
            fig.add_trace(go.Scatter(y=np.imag(data), name="data imag", mode="lines+markers"), row=1, col=1)

            fig.add_trace(go.Scatter(y=np.real(x1), name="x1 real", mode="lines+markers"), row=2, col=1)
            fig.add_trace(go.Scatter(y=np.imag(x1), name="x1 imag", mode="lines+markers"), row=2, col=1)
            fig.update_layout(title="upsample data")
            pyo.plot(fig, filename="data_shape_upsample_plot.html")
        return x1
        # some debug code
        #       x1[mm+kk]=np.dot(reg,(hh2[kk,:]).T)   # x1[mm+kk]=reg*hh2[kk:]'
        #       x1i[mm+kk]=np.dot(np.real(reg),(hh2[kk,:]).T)
        #       x1q[mm+kk]=np.dot(np.imag(reg),(hh2[kk,:]).T)
        #   mm=mm+8

        # add channel noise, can set noise to zero here
        # sd=0.001/sqrt(2);
        # sd=0.5/sqrt(2);
        # x2=x1+sd*(rand(1,8*N_dat)+j*rand(1,8*N_dat));
        #
        # create a frequency and phase offset
        # spin sampled received signal from channel
        # x3=x2.*exp(j*2*pi*(1:8*N_dat)*ff+j*phs_0);
        #
        # np array [start:stop:step]
        # x4=x3(7:4:8*N_dat);             % downsample to 2-samples per symbol
        #                                % offset 1 sample for timing loop to track

    def slider_plot(self, plots, fname):
        """
        :param plots:  list of 1d items to plot
        :return:
        """
        # Create figure -----------------------------------------
        fig = go.Figure()
        # Add traces, one for each slider step
        for step in range(len(plots)):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    mode="lines+markers",
                    line=dict(color="#00CED1", width=2),
                    marker=dict(color="red", size=5),
                    name="index = " + str(step),
                    x=list(range(len(plots[0]))),
                    y=plots[step])
            )

        # Make 10th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "MF path: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "index: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )
        pyo.plot(fig, filename=fname)
        return

    def compare_IQ_plot(self, x0, x1, x0name, x1name):
        fig = go.Figure().set_subplots(2, 2)
        # fig = make_subplots(rows=2, cols=2)
        fig.add_trace(go.Scatter(y=np.real(x0), name=x0name+" real",
                                 mode="lines+markers", marker=dict(color="red", size=5),
                                 line=dict(width=2, color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(y=np.imag(x0), name=x0name+" imag",
                                 mode="lines+markers", marker=dict(color="red", size=5),
                                 line=dict(width=2, color="green")), row=1, col=2)
        fig.add_trace(go.Scatter(y=np.real(x1), name=x1name+" real",
                                 mode="lines+markers", marker=dict(color="red", size=5),
                                 line=dict(width=2, color="cyan")), row=2, col=1)
        fig.add_trace(go.Scatter(y=np.imag(x1), name=x1name+" imag",
                                 mode="lines+markers", marker=dict(color="red", size=5),
                                 line=dict(width=2, color="#404040")), row=2, col=2)
        fig.update_layout(title=x0name + " vs " + x1name)
        pyo.plot(fig,  filename="compare_IQ_timeseries_plot.html")

    def get_data_filter_paths(self, x0, hh):
        fpaths = hh.shape[0]
        flength = hh.shape[1]
        y0 = np.zeros(len(x0), dtype='complex')
        plots = [0] * fpaths
        for kk in range(fpaths):
            shiftreg = np.zeros(flength, dtype='complex')
            for nn in range(len(x0)):
                shiftreg = np.append(x0[nn], shiftreg[0:-1])
                y0[nn] = shiftreg.dot(hh[kk].conj())
            plots[kk] = y0.copy()
        return plots

    def matched_filter_test(self, x4, mf, dmf):
        fpaths = mf.shape[0]
        flength = mf.shape[1]
        nplots = fpaths
        ncol = 5
        x5  = lambda m: sig.lfilter(mf[m, :], 1, x4)
        x5d = lambda m: sig.lfilter(dmf[m, :], 1, x4)
        ncol = 5
        nrow = int(np.ceil(nplots / ncol))

        if not USEMFANIMATION:
            # Create figure -----------------------------------------
            fig = go.Figure()
            # Add traces, one for each slider step
            for step in range(fpaths):
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        mode="lines+markers",
                        line=dict(color="#00CED1", width=2),
                        marker=dict(color="red", size=5),
                        name="MF = " + str(step),
                        x=list(range(100, 200)),
                        y=np.real(x5(step)[100:200]))
                )

            # Make 10th trace visible
            fig.data[0].visible = True

            # Create and add slider
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)},
                          {"title": "MF path: " + str(i)}],  # layout attribute
                )
                step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
                steps.append(step)

            sliders = [dict(
                active=0,
                currentvalue={"prefix": "MF: "},
                pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                sliders=sliders
            )
            pyo.plot(fig, filename="matched_filter_slider_plot.html")
        else:
            # create figure animation
            fig = go.Figure(
                data=[go.Scatter(x=list(range(100, 200)), y=np.real(x5(0)[100:200]),
                                 mode="lines",
                                 line=dict(width=2, color="green"))],
                layout=go.Layout(
                    xaxis=dict(range=[100, 200], autorange=False, zeroline=False),
                    yaxis=dict(range=[-1.5, 1.5], autorange=False, zeroline=False),
                    title_text="matched filter test", hovermode="closest",
                    updatemenus=[dict(type="buttons",
                                      buttons=[dict(label="Play",
                                                    method="animate",
                                                    args=[None])])]),
                frames=[go.Frame(
                    data=[go.Scatter(x=list(range(100, 200)), y=np.real(x5(k)[100:200]),
                                mode="lines+markers",
                                marker=dict(color="red", size=5),
                                line=dict(width=2, color="green"))])
                    for k in range(fpaths)]
            )
            pyo.plot(fig, filename="matched_filter_plot.html")
        print('end')


        # <remove later>
        # % text(5, 1.4, ['Filter Set (', num2str(m, '%4.0f%'), ')'], 'fontsize', 14)
        # text(5, 1.8, ['Filter Set (', num2str(m), ')'], 'fontsize', 14)
        #
        # title('Time Response, Matched Filter Response Succesive Weight Sets', 'fontsize', 14)
        # xlabel('Time index', 'fontsize', 14)
        # ylabel('Amplitude', 'fontsize', 14)
        # axis([0 200 - 2 2])
        # pause(0.4)
        # end

    def timing_loop(self, x4, hh2, dhh2):
        """
        ---- Timing Loop -----------------------------------------------
        Timing Recovery Loop
        see Advanced_Topics_8_SynchronizationTechniques.ppt, Fred Harris
        """
        # theta_0 = 2*np.pi/200
        theta_0 = 2 * np.pi / 200
        eta = np.sqrt(2)/2
        eta = 6 * eta       # was 6

        k_i_t = (4 * (theta_0 ** 2))  / (1 + (2 * eta * theta_0) + (theta_0 ** 2))
        k_p_t = (4 * eta * theta_0) / (1 + (2 * eta * theta_0) + (theta_0 ** 2))

        fpaths = hh2.shape[0]  # number of filter paths
        flength = hh2.shape[1]     # each path filter length
        reg_t = np.zeros(flength, dtype='complex')
        int_t = 0.0
        accum_t = self.timing_idx_start
        ndata = len(x4)
        self.accum_t_sv = np.zeros(ndata)
        self.det_t_sv = np.zeros(ndata)
        x6 = np.zeros(len(x4), dtype='complex')
        mm = 0                                       # output clock at 1-sample per symbol
        os = 0                                       # offset for bit slip
        for nn in range(0, len(x4)-2, 2):            # =1:2:length(x4)-2
            pntr = int(np.floor(accum_t))            # point to a coefficient set
            reg_t = np.append(x4[nn+os], reg_t[0:-1])   # shift register, new in from left
            y_t1 = reg_t.dot(hh2[pntr, :].conj())    # polyphase Matched Filter output time sample
            dy_t1 = reg_t.dot(dhh2[pntr, :].conj())  # derivative Matched Filter output time sample
            x6[nn] = y_t1                            # save MF output sample

            reg_t = np.append(x4[nn+os+1], reg_t[0:-1]) # new sample in matched filter register
            y_t2 = reg_t.dot(hh2[pntr, :].conj())    # MF output time sample
            dy_t2 = reg_t.dot(dhh2[pntr, :].conj())  # dMF output time sample
            x6[nn + 1] = y_t2

            det_t = np.real(y_t1) * np.real(dy_t1)   # y*dy product (timing error)
            self.det_t_sv[mm] = det_t                # save timing error
            int_t = int_t + (k_i_t * det_t)          # Loop filter integrator
            sm_t = int_t + (k_p_t * det_t)           # loop filter output

            self.accum_t_sv[mm] = accum_t            # save timing accumulator content
            mm += 1                                  # increment symbol clock
            accum_t = accum_t + sm_t                 # update accumulator
            if accum_t > 33:                       # test for accumulator overflow
                if os < 1:
                    os += 1
            if accum_t < 0:                         # test for accumulator underflow
                if os > -1:
                    os -= 1                         # offset either -1, 0 +1
            accum_t = accum_t % fpaths              # return to zero once accum t gets past fpaths

            # if nn > 90:
            #     print('pause')
        if DEBUGTIMING:
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(y=self.accum_t_sv, name="accum_t", mode="lines+markers"), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.det_t_sv, name="det_t_sv", mode="lines+markers"), row=2, col=1)
            fig.update_layout(title="timing loop, accumulator, timing error")
            pyo.plot(fig, filename="timing_loop_plot.html")
        return x6

    def phase_lock_loop(self, x6):
        # ---- phase lock loop --------------------

        theta_0 = 2 * np.pi / 500
        eta = np.sqrt(2) / 2
        
        theta_0 = theta_0 / 10
        eta = eta / 10

        k_i_ph = (4 * (theta_0 ** 2)) / (1 + (2 * eta * theta_0) + (theta_0 ** 2)) # 0.000620
        k_p_ph = (4 * eta * theta_0) / (1 + (2 * eta * theta_0) + (theta_0 ** 2))  # ~0.0349170

        # k_i_ph = 0.0
        # k_p_ph = 0.001


        int_ph = 0.0
        accum_ph = 0.0
        lp_flt = 0.0

        phs_err_sv = np.zeros(len(x6))
        accum_ph_sv = np.zeros(len(x6))
        x7 = np.zeros(len(x6), dtype='complex')

        # mm=0
        for nn in range(len(x6)):
            prod = x6[nn] * np.exp(1j * 2 * np.pi * accum_ph)   # << replace with cordic or equivalent
            x7[nn] = prod

            if nn % 2 == 0:  # rem(nn,2)==1:
                det_ph = np.sign(np.real(prod)) + (1j * np.sign(np.imag(prod)))
                phs_err = np.angle(det_ph * prod.conj()) / (2 * np.pi)
                phs_err_sv[nn] = phs_err
                int_ph = int_ph + k_i_ph * phs_err
                lp_flt = int_ph + k_p_ph * phs_err
                # mm += 1

            accum_ph_sv[nn] = accum_ph
            accum_ph = accum_ph + lp_flt
            # if nn == 2000:
            #     print('pause')
            #     input('pause')

        if DEBUGPLL:
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(y=accum_ph_sv, name="accum_t", mode="lines+markers"),
                          #go.Layout(yaxis=dict(range=[-1.5, 1.5], autorange=False, zeroline=False)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(y=phs_err_sv, name="det_t_sv", mode="lines+markers"), row=2, col=1)
            fig.update_layout(title="PLL, accumulator, phase error")
            pyo.plot(fig, filename="PLL_accum_err_plot.html")
            md.compare_IQ_plot(x6, x7, "x6 pre pll", "x7 post pll")
            plt_start = -310
            fig = go.Figure().set_subplots(1, 2)  # make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=np.real(x7[plt_start:-10:2]), y=np.imag(x7[plt_start:-10:2]),
                                     name="IQ even", mode="markers"), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.real(x7[plt_start+1:-11:2]), y=np.imag(x7[plt_start+1:-11:2]),
                                     name="IQ odd", mode="markers"), row=1, col=2)
            pyo.plot(fig, filename="x7_IQ_PLL_constellation_plot.html")
        return x7

if __name__ == '__main__':

    DEBUGDERIVATIVEFILTER = True
    DEBUGUPSAMPLE = True
    DEBUGTIMING = True
    DEBUGPLL = True
    ADDIMPAIRMENTS = False
    SNR = 20
    USEMFANIMATION = False
    # frequency is normalized to 1 or for example 1MHz
    ff = 0.000020  # equivalent to 20KHz from 1MHz or 0.002%
    # %ff=0.00002           # frequency offset must be small
    # %ff=0.0               # frequency offset equal 0
    phs_0 = 0.05 * 2 * np.pi  # phase offset
    # phs_0 =0              # phase offset equal to zero
    # mod_flg = 'qpsk'  # mod_flg 'qpsk' or 'qam'
    n_data = 4000
    
    ndx_strt = 10  # starting index of timing pointer

    # instace class opbject and setup params
    md = ModulateDemodulate()
    vis = visualization.Visualization()
    md.modulation = '16apsk'  # one of these: 'qpsk' '8psk' 'qpsk'  '16apsk' '32apsk'
    md.code_rate = 'c3/4'

    # data: first generate the data
    iq_list = md.get_constellation()  # generate a constellation list for the 
    
    # >> debug for diff constellations <<<
    x0 = md.generate_data(n_data=n_data, iq_list=iq_list)


    vis.plot_constellation(x0, -210, -10, 'x0')

    # shaping filter
    hh, hh2 = md.polyphase_filter(over_sample=8, alpha=0.5, n_symbol=6)
    vis.plot_data([hh,hh2[0,:]], ['hh-shaping filter', 'hh2 0'])

    # timing filter
    mf, dmf = md.polyphase_derivative_filter(n_paths=32, over_sample=64,
                                             alpha=0.5, n_symbol=6)

    # upsample data as if it was going to be transmitted.
    # upsample to transmitted sample rate for simulation purposes. then add channel impairments
    # x0 = md.generate_data(n_data=n_data, modulation='qpsk')
    x1 = md.shape_upsample(hh2=hh2, data=x0)
    if ADDIMPAIRMENTS:
        x2 = awgn(x1, SNR)
        x3 = add_frequency_offset(x2, 1.0, 0.000020, np.pi/2)  # phase offset in radians
    else:
        x3 = x1

    x4 = x3[1::4]  # shift by 0 or 1 then sample every nth

    md.slider_plot(np.real(md.get_data_filter_paths(x4[100:200], mf)), "x4_data_polyphase_filtered.html")
    md.matched_filter_test(x4, mf, dmf)

    # timing loop, freq locked loop
    x6 = md.timing_loop(x4, mf, dmf)

    # real and imag signals time series plot
    md.compare_IQ_plot(x4,x6, "x4 pre", "x6 post")

    fig = go.Figure().set_subplots(1, 1)  # make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.real(x6[-1000:-100:2]), y=np.imag(x6[-1000:-100:2]), mode="markers"))
    pyo.plot(fig, filename="x6_IQ_constellation_plot.html")

    # Phase locked loop for carrier recovery. 
    md.phase_lock_loop(x6)

    print('end')
    # modulator (transmitter)

    over_sample = 8  # 8-samples per symbol at transmitter,
    # % downsampled to 2-samples at receiver
    n_symbol = 6  # % filter width / 2
    alpha = 0.5


    # -----------------
    """
    plt.figure(1);
    # subplot(2,2,1);
    plt.plot(real(x1), 'r.')
    plt.grid(True)

    # plt.axis(['square']);

    plt.title('Constellation 8-Samples per Symbol');

    # %%

    subplot(2, 2, 2)
    plot(x1, 'b.');
    axis(['square']);
    grid
    on;
    title('Constellation 8-Samples per Symbol');

    subplot(2, 2, 4)
    plot(x2, 'b.');
    axis(['square']);
    grid
    on;
    title('Constellation 8-Samples per Symbol');

    subplot(2, 2, 3)
    plot(0, 0)
    hold
    on
    for k=1:16: (N_dat * 4) - 16
    plot(-1: 1 / 8:1, real(x2(k: k + 16)))
    plot(-1: 1 / 8:1, real(x2(k: k + 16)), 'r.')
    end
    hold
    off
    grid
    on
    title('Eye Diagram, 8-Samples per Symbol, No Offset Relative to Index 0')

    subplot(2, 2, 2)
    plot(x4(1: 2:N_dat), 'r.', 'linewidth', 2
    ')
    axis(['square'])
    grid
    on
    title('Constellation 2-Samples per Symbol')

    subplot(2, 2, 4)
    plot(0, 0)
    hold
    on
    for k=1:4: N_dat - 4
    plot(-1: 1 / 2:1, real(x4(k: k + 4)))
    plot(-1: 1 / 2:1, real(x4(k: k + 4)), '.r')
    end
    hold
    off
    grid
    on
    title('Eye Diagram, 2-Samples per Symbol, Offset Relative to Index 0')

    # %pause

    # % matched filter test
    figure(2)
    for m=1:32
    x5 = filter(hh_t2(m,:), 1, x4);
    plot(1: 200, real(x5(1: 200)))
    hold
    on
    plot(1: 2:200, real(x5(1: 2:200)), 'ro', 'linewidth', 2)
    hold
    off
    grid
    on
    # %  text(5,1.4,['Filter Set (',num2str(m,'%4.0f%'),')'],'fontsize',14)
    text(5, 1.8, ['Filter Set (', num2str(m), ')'], 'fontsize', 14)

    title('Time Response, Matched Filter Response Succesive Weight Sets', 'fontsize', 14)
    xlabel('Time index', 'fontsize', 14)
    ylabel('Amplitude', 'fontsize', 14)
    axis([0 200 - 2 2])
    pause(0.4)
    end

    # %pause
    figure(3)
    for m=1:32
    x5 = filter(hh_t2(m,:), 1, x4);
    subplot(2, 1, 1)
    plot(x5(1: 2:N_dat), 'r.', 'linewidth', 2)
    grid
    on
    axis('square')
    axis([-1.5 1.5 - 1.5 1.5])
    % text(2.0, 0.2, ['Filter Set (', num2str(m, '%4.0f%'), ')'], 'fontsize', 14)
    text(2.0, 0.2, ['Filter Set (', num2str(m), ')'], 'fontsize', 14)
    title('Constellation Successive MF Weight Sets', 'fontsize', 14)

    subplot(2, 1, 2)
    plot(0, 0)
    hold
    on
    for k=1:4: N_dat - 4
    plot(-1: 1 / 2:1, real(x5(k: k + 4)))
    plot(-1: 1, real(x5(k: 2:k + 4)), 'r.', 'linewidth', 2)
    end
    hold
    off
    grid
    on
    title('Eye Diagram Successive Weight Sets', 'fontsize', 14
    ')
    # %pause(0.4)
    end



    figure(5)
    subplot(2,1,1)
    plot(accum_t_sv(1:N_dat))
    hold on
    plot(floor(accum_t_sv(1:N_dat)),'r')
    hold off
    grid on
    axis([0 2000 0 32])
    title('Timing Loop Phase Accumulator and Pointer')

    subplot(2,2,3)
    plot(0,0)
    hold on
    for nn=1001:4:length(x6)-4:
        plot(-1:1/2:1,real(x6(nn:nn+4)))
    end
    hold off
    grid on
    title('Eye Diagram')

    subplot(2,2,4)
    plot(x6(501:2:length(x6)),'.')
    hold on
    plot(x6(1001:2:length(x6)),'r.')

    hold off
    grid on
    axis('square')
    title('Constellation Diagram')


    figure(6)
    subplot(2, 1, 1)
    plot(phs_err_sv(1: N_dat - 20))
    grid
    on
    title('Phase Lock Loop Phase Error')

    subplot(2, 2, 3)
    plot(0, 0)
    hold
    on
    for nn in range(501, len(x7) - 4, 4):
        plot(-1: 1 / 2:1, real(x7(nn: nn + 4)))

    hold
    off
    grid
    on
    title('Eye Diagram After Phase PLL')

    subplot(2, 2, 4)
    plot(x7(1: 2:N_dat), '.')
    hold
    on
    % plot(x7(201: 2:2000), '.r')
    plot(x7(351: 2:N_dat - 20), '.r')
    hold
    off
    grid
    on
    axis('square')
    axis([-1.5 1.5 - 1.5 1.5])
    title('Constellation Diagram')
    """
    # x = np.array(1.60862725407784 - 0.138377852614227i,0.812640904023591 - 0.0166684523293880i,
    #                 -0.992334783646098 + 0.222420229317273i,-1.38503577354905 - 0.191155436978425i,
    #                 -0.263568595822458 - 1.00419927039655i,0.192155659749720 - 1.41247809044304i,
    #                 0.0614580660094335 - 1.41331088496382i,   -0.233745830649673 - 1.01654136275887i,
    #                 -1.01951371772027 - 0.147275724879778i,   -1.45499249170421 + 0.206182789749751i,
    #                 -1.18996310300382 + 0.0488912417679827i, -1.22144963019154 + 0.0485637457532829i,
    #                 -1.43402709368722 - 0.147820788388406i, -1.01697216909647 + 0.214812815520908i,
    #                 -0.162292316648681 + 1.39063576460403i,   0.216303205566145 + 1.01662769240451i,
    #                 -0.0478901141969781 - 0.862978460382268i, -0.118053264757420 - 1.57101408004009i,
    #                 0.155890423149116 - 1.39428001452904i, -0.244901822017986 - 1.04062368619725i,
    #                 -1.22323550792338 + 0.0562844985734103i, -1.19169155415990 + 0.0576665833148051i,
    #                 -0.270328779891253 - 1.40939385784050i,   0.155413094266565 - 1.02251109682525i,
    #                 0.0284182353726293 + 0.830313027038819i,0.0371260287599868 + 1.55774021769522i])

