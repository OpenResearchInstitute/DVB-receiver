# Polyphase Filter RTL Module

A polyphase filter engine to perform filtered rate change in an efficient manner.  Designed to form part of a DVB-S2X transmitter.

## Architecture

### Filter Coefficients

The filter coefficients are loaded through a AXI-stream bus port.  In normal operation this will be loaded from a processor such as a Zynq's ARM core.

## TODO

- Decimation is not currently implemented
- With a higher clock to sample rate ratio a more efficient implementation is possible by increasing time sharing of resources
