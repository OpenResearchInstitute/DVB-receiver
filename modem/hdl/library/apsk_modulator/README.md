# APSK (Amplitude Phase Shift Keying) Modulator RTL Module

A generic APSK modulator engine, supporting one to eight bits per sample.  Designed to form part of a DVB-S2X transmitter.

## Architecture

A diagram of the APSK modulator architecture is shown below:

![APSK Modulator Architecture](doc/block_diagram.svg)

The design currently instantiates Xilinx primitive macros although these could be easily changed to another architecture (FPGA or ASIC).
