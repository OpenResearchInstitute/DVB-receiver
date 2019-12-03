# DVB-S2X MODEM HDL Top Level

This folder contains the HDL description of the modem with associated tests.

## Module List

- [APSK Modulator](library/apsk_modulator)
- [Polyphase Filter](library/polyphase_filter)
- [FIR Filter](library/fir_filter)
- [Multiply Accumulate](library/multiply_accumulate)
- [Lookup Table](library/lookup_table)

## Testing

All modules use the [CocoTB](https://github.com/cocotb/cocotb) library for high level test bench creation.  Each module has unit tests to perform regression testing following changes.

### Setup

#### System Installs

Install the required simulation tools on a Ubuntu/Debian distribution:

    sudo apt-get install iverilog gtkwave

#### CocoTB Installation

A Python virtual enviroment is used to manage and distribute the Python enviroment.  To set this up run the `setup_venv.sh` script in the [python](../python) folder.  

Following succesful virtual enviroment setup all the tests can be run using the `rull_all_tests.sh` script in the [library](library) folder.

Individual module tests can be performed by using the `run_test.ssh` in each modules `test` folder.  
