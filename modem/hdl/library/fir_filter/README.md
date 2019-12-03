# FIR Filter RTL Module

A FIR filter engine.  Designed to form part of a DVB-S2X transmitter.

## Architecture

### Filter Coefficients

The filter coefficients are loaded through a AXI-stream bus port.  In normal operation this will be loaded from a processor such as a Zynq's ARM core.

## TODO

- The option to perform time sharing of the DSP48 slice would increase resource efficiency with high clock to sample rate ratio.
