# DVB-S2X MODEM Python Definitions

A collection of Python code which is used to model, simulate and verify the modem implementation.

## File Breakdown

### [algorithm](algorithm)

Scripts used for algortihm development such as [DVB-S2X modulation plot](algorithm/modulator/DVB-S2X Modulation Generation.ipynb).


### [library](library)

A number of library files which are used across the modem design for modelling and verification.  Notable files include:

- [DVB-S2X Constellation Generator](library/generate_dvb-s2x_constellations.py)
- [DVB-S2X Constellation Definition](library/DVB-S2X_constellations.json)
- [Generic Modulator](library/generic_modem.py)



### [cocotb](cocotb)

Additional or modified CocoTB drivers used during HDL verification.
