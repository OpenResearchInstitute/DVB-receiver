# MODEM (Modulator / Demodulator) for DVB-S2X Implementation

This folder contains the modulator and demodulator of the DVB-S2X implementation with high level Python code, HDL and verification.

## Module List

- [APSK Modulator](hdl/library/apsk_modulator)
- [Polyphase Filter](hdl/library/polyphase_filter)
- [FIR Filter](hdl/library/fir_filter)
- [Multiply Accumulate](hdl/library/multiply_accumulate)
- [Lookup Table](hdl/library/lookup_table)

## Testing

All modules use the [CocoTB](https://github.com/cocotb/cocotb) library for high level test bench creation.  Each module has unit tests to perform regression testing following changes.

### Setup

A Python virtual enviroment is used to manage and distribute the Python enviroment.  To set this up run the `setup_venv.sh` script in the [python](../python) folder.  

Following succesful virtual enviroment setup all the tests can be run using the `rull_all_tests.sh` script in the [hdl/library](hdl/library) folder.

Individual module tests can be performed by using the `rull_testssh` in each modules `test` folder.  