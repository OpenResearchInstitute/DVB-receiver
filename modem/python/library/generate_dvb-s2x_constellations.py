import numpy as np
import json

# the name of the JSON output file
json_filename = "DVB-S2X_constellations.json"


# create constellation data structure
constellations = {"modulation_type_depth" : 2}


## pi/2 BPSK MODCODS
bits_per_symbol = 1
bit_map_cartesian = [   [1.0/np.sqrt(2), 1.0/np.sqrt(2)],
						[-1.0/np.sqrt(2), -1.0/np.sqrt(2)]]

# create dictionary
mod = { "bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["129"] = {	"BPSK 1/5"		:	mod,
							"BPSK 11/45"	:	mod,
							"BPSK 1/3"		:	mod,
							"BPSK-S 1/5"	:	mod,
							"BPSK-S 11/45"	:	mod}

constellations["131"] = {	"BPSK 1/5"		:	mod,
							"BPSK 4/15"		:	mod,
							"BPSK 1/3"		:	mod}

## QPSK
bits_per_symbol = 2
bit_map_cartesian = [   [np.sqrt(2), np.sqrt(2)],
						[-np.sqrt(2), np.sqrt(2)],
						[np.sqrt(2), -np.sqrt(2)],
						[-np.sqrt(2), -np.sqrt(2)]]

# create dictionary
mod = {"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["129"]["QPSK 2/9"] = mod
constellations["132"] = {"QPSK 13/45" : mod}
constellations["134"] = {"QPSK 9/20" : mod}
constellations["136"] = {"QPSK 11/20" : mod}
constellations["216"] = {"QPSK 11/45" : mod}
constellations["218"] = {"QPSK 4/15" : mod}
constellations["220"] = {"QPSK 14/45" : mod}
constellations["222"] = {"QPSK 7/15" : mod}
constellations["224"] = {"QPSK 8/15" : mod}
constellations["226"] = {"QPSK 32/45" : mod}


## 8PSK
bits_per_symbol = 3

# set the bit mapping table
bit_map_phasor =   [[1, np.pi/4],
					[1, 0],
					[1, 4*np.pi/4],
					[1, 5*np.pi/4],
					[1, 2*np.pi/4],
					[1, 7*np.pi/4],
					[1, 3*np.pi/4],
					[1, 6*np.pi/4]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["142"] = {"8PSK 23/36" : mod}
constellations["144"] = {"8PSK 25/36" : mod}
constellations["146"] = {"8PSK 13/18" : mod}
constellations["228"] = {"8PSK 7/15" : mod}
constellations["230"] = {"8PSK 8/15" : mod}
constellations["232"] = {"8PSK 26/45" : mod}
constellations["234"] = {"8PSK 32/45" : mod}

## 8PSK - 100/180
bits_per_symbol = 3

# set the bit mapping table
R1 = 1.0/6.8
R2 = 5.32/6.8
bit_map_phasor =   [[R1, 0],
					[R2, 1.352*np.pi],
					[R2, 0.648*np.pi],
					[1.0, 0],
					[R1, np.pi],
					[R2, -0.352*np.pi],
					[R2, 0.352*np.pi],
					[1.0, np.pi]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["138"] = {"8APSK 5/9-L" : mod}


## 8PSK - 104/180
bits_per_symbol = 3

# set the bit mapping table
R1 = 1.0/8.0
R2 = 6.39/8.0
bit_map_phasor =   [[R1, 0],
					[R2, 1.352*np.pi],
					[R2, 0.648*np.pi],
					[1.0, 0],
					[R1, np.pi],
					[R2, -0.352*np.pi],
					[R2, 0.352*np.pi],
					[1.0, np.pi]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["140"] = {"8APSK 26/45-L" : mod}


## 16APSK - 90/180, 96/180, 100/180
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.7
bit_map_phasor =   [[R1, 1*np.pi/8],
					[R1, 3*np.pi/8],
					[R1, 7*np.pi/8],
					[R1, 5*np.pi/8],
					[R1, 15*np.pi/8],
					[R1, 13*np.pi/8],
					[R1, 9*np.pi/8],
					[R1, 11*np.pi/8],
					[1.0, 1*np.pi/8],
					[1.0, 3*np.pi/8],
					[1.0, 7*np.pi/8],
					[1.0, 5*np.pi/8],
					[1.0, 15*np.pi/8],
					[1.0, 13*np.pi/8],
					[1.0, 9*np.pi/8],
					[1.0, 11*np.pi/8]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["148"] = {"16APSK 1/2-L" : mod}
constellations["150"] = {"16APSK 8/15-L" : mod}
constellations["152"] = {"16APSK 5/9-L" : mod}



## 16PSK - 26/45, 3/5
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.7
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["154"] = {"16APSK 26/45" : mod}
constellations["156"] = {"16APSK 26/45" : mod}
constellations["240"] = {"16APSK 26/45" : mod}
constellations["242"] = {"16APSK 3/5" : mod}



## 16PSK - 18/30
bits_per_symbol = 4

# set the bit mapping table
bit_map_cartesian =    [[0.4718, 0.2606],
						[0.2606, 0.4718],
						[-0.4718, 0.2606],
						[-0.2606, 0.4718],
						[0.4718, 0.2606],
						[0.2606, 0.4718],
						[-0.4718, 0.2606],
						[-0.2606, 0.4718],
						[1.2088, 0.4984],
						[0.4984, 1.2088],
						[-1.2088, 0.4984],
						[-0.4984, 1.2088],
						[1.2088, 0.4984],
						[0.4984, 1.2088],
						[-1.2088, 0.4984],
						[-0.4984, -1.2088]]

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["158"] = {"16APSK 3/5-L" : mod}



## 16PSK - 28/45, 8/15
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.5
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["160"] = {"16APSK 28/45" : mod}
constellations["238"] = {"16APSK 8/15" : mod}



## 16APSK - 23/36, 25/36, 
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.1
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["162"] = {"16APSK 23/36" : mod}
constellations["166"] = {"16APSK 25/36" : mod}

## 16PSK - 20/30
bits_per_symbol = 4

# set the bit mapping table
bit_map_cartesian =    [[0.5061, 0.2474],
						[0.2474, 0.5061],
						[-0.5061, 0.2474],
						[-0.2474, 0.5061],
						[0.5061, 0.2474],
						[0.2474, 0.5061],
						[-0.5061, 0.2474],
						[-0.2474, 0.5061],
						[1.2007, 0.4909],
						[0.4909, 1.2007],
						[-1.2007, 0.4909],
						[-0.4909, 1.2007],
						[1.2007, 0.4909],
						[0.4909, 1.2007],
						[-1.2007, 0.4909],
						[-0.4909, 1.2007]]

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["164"] = {"16APSK 2/3-L" : mod}




## 16PSK - 13/18, 32/45
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/2.85
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["168"] = {"16APSK 13/18" : mod}
constellations["244"] = {"16APSK 32/45" : mod}



## 16PSK - 140/180
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.6
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["170"] = {"16APSK 7/9" : mod}



## 16PSK - 154/180
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.2
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["172"] = {"16APSK 77/90" : mod}




## 16PSK - 7/15
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.32
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["236"] = {"16APSK 7/15" : mod}




## 16PSK - 8/15
bits_per_symbol = 4

# set the bit mapping table
R1 = 1.0/3.5
bit_map_phasor =   [[1.0,	3*np.pi/12],
					[1.0,	21*np.pi/12],
					[1.0,	9*np.pi/12],
					[1.0,	15*np.pi/12],
					[1.0,	1*np.pi/12],
					[1.0,	23*np.pi/12],
					[1.0,	11*np.pi/12],
					[1.0,	13*np.pi/12],
					[1.0,	5*np.pi/12],
					[1.0,	19*np.pi/12],
					[1.0,	7*np.pi/12],
					[1.0,	17*np.pi/12],
					[R1, 	3*np.pi/12],
					[R1, 	21*np.pi/12],
					[R1, 	9*np.pi/12],
					[R1, 	15*np.pi/12]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["238"] = {"16APSK 8/15" : mod}




## 32APSK - 2/3
bits_per_symbol = 5

# set the bit mapping table
R1 = 1.0/5.55
R2 = 2.85/5.55

bit_map_phasor =   [[1,	11*np.pi/16],
					[1,	9*np.pi/16],
					[1,	5*np.pi/16],
					[1,	7*np.pi/16],
					[R2, 9*np.pi/12],
					[R2, 7*np.pi/12],
					[R2, 3*np.pi/12],
					[R2, 5*np.pi/12],
					[1,	13*np.pi/16],
					[1,	15*np.pi/16],
					[1,	3*np.pi/16],
					[1,	1*np.pi/16],
					[R2, 11*np.pi/12],
					[R1, 3*np.pi/4],
					[R2, 1*np.pi/12],
					[R1, 1*np.pi/4],
					[1,	21*np.pi/16],
					[1,	23*np.pi/16],
					[1,	27*np.pi/16],
					[1,	25*np.pi/16],
					[R2, 15*np.pi/12],
					[R2, 17*np.pi/12],
					[R2, 21*np.pi/12],
					[R2, 19*np.pi/12],
					[1,	19*np.pi/16],
					[1,	17*np.pi/16],
					[1,	29*np.pi/16],
					[1,	31*np.pi/16],
					[R2, 13*np.pi/12],
					[R1, 5*np.pi/4],
					[R2, 23*np.pi/12],
					[R1, 7*np.pi/4]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["174"] = {"32APSK 2/3-L" : mod}
constellations["246"] = {"32APSK 2/3" : mod}



## 32APSK - 128/180
bits_per_symbol = 5

# set the bit mapping table
R1 = 1.0/5.6
R2 = 2.6/5.6
R3 = 2.99/5.6

bit_map_phasor =   [[R1, 1*np.pi/4],
					[1.0, 7*np.pi/16],
					[R1, 7*np.pi/4],
					[1.0, 25*np.pi/16],
					[R1, 3*np.pi/4],
					[1.0, 9*np.pi/16],
					[R1, 5*np.pi/4],
					[1.0, 23*np.pi/16],
					[R2, 1*np.pi/12],
					[1.0, 1*np.pi/16],
					[R2, 23*np.pi/12],
					[1.0, 31*np.pi/16],
					[R2, 11*np.pi/12],
					[1.0, 15*np.pi/16],
					[R2, 13*np.pi/12],
					[1.0, 17*np.pi/16],
					[R2, 5*np.pi/12],
					[1.0, 5*np.pi/16],
					[R2, 19*np.pi/12],
					[1.0, 27*np.pi/16],
					[R2, 7*np.pi/12],
					[1.0, 11*np.pi/16],
					[R2, 17*np.pi/12],
					[1.0, 21*np.pi/16],
					[R3, 1*np.pi/4],
					[1.0, 3*np.pi/16],
					[R3, 7*np.pi/4],
					[1.0, 29*np.pi/16],
					[R3, 3*np.pi/4],
					[1.0, 13*np.pi/16],
					[R3, 5*np.pi/4],
					[1.0, 19*np.pi/16]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["178"] = {"32APSK 32/45" : mod}



## 32APSK - 132/180
bits_per_symbol = 5

# set the bit mapping table
R1 = 1/5.6
R2 = 2.6/5.6
R3 = 2.86/5.6

bit_map_phasor =   [[R1, 1*np.pi/4],
					[1.0, 7*np.pi/16],
					[R1, 7*np.pi/4],
					[1.0, 25*np.pi/16],
					[R1, 3*np.pi/4],
					[1.0, 9*np.pi/16],
					[R1, 5*np.pi/4],
					[1.0, 23*np.pi/16],
					[R2, 1*np.pi/12],
					[1.0, 1*np.pi/16],
					[R2, 23*np.pi/12],
					[1.0, 31*np.pi/16],
					[R2, 11*np.pi/12],
					[1.0, 15*np.pi/16],
					[R2, 13*np.pi/12],
					[1.0, 17*np.pi/16],
					[R2, 5*np.pi/12],
					[1.0, 5*np.pi/16],
					[R2, 19*np.pi/12],
					[1.0, 27*np.pi/16],
					[R2, 7*np.pi/12],
					[1.0, 11*np.pi/16],
					[R2, 17*np.pi/12],
					[1.0, 21*np.pi/16],
					[R3, 1*np.pi/4],
					[1.0, 3*np.pi/16],
					[R3, 7*np.pi/4],
					[1.0, 29*np.pi/16],
					[R3, 3*np.pi/4],
					[1.0, 13*np.pi/16],
					[R3, 5*np.pi/4],
					[1.0, 19*np.pi/16]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["180"] = {"32APSK 11/15" : mod}




## 32APSK - 140/180
bits_per_symbol = 5

# set the bit mapping table
R1 = 1/5.6
R2 = 2.8/5.6
R3 = 3.08/5.6

bit_map_phasor =   [[R1, 1*np.pi/4],
					[1.0, 7*np.pi/16],
					[R1, 7*np.pi/4],
					[1.0, 25*np.pi/16],
					[R1, 3*np.pi/4],
					[1.0, 9*np.pi/16],
					[R1, 5*np.pi/4],
					[1.0, 23*np.pi/16],
					[R2, 1*np.pi/12],
					[1.0, 1*np.pi/16],
					[R2, 23*np.pi/12],
					[1.0, 31*np.pi/16],
					[R2, 11*np.pi/12],
					[1.0, 15*np.pi/16],
					[R2, 13*np.pi/12],
					[1.0, 17*np.pi/16],
					[R2, 5*np.pi/12],
					[1.0, 5*np.pi/16],
					[R2, 19*np.pi/12],
					[1.0, 27*np.pi/16],
					[R2, 7*np.pi/12],
					[1.0, 11*np.pi/16],
					[R2, 17*np.pi/12],
					[1.0, 21*np.pi/16],
					[R3, 1*np.pi/4],
					[1.0, 3*np.pi/16],
					[R3, 7*np.pi/4],
					[1.0, 29*np.pi/16],
					[R3, 3*np.pi/4],
					[1.0, 13*np.pi/16],
					[R3, 5*np.pi/4],
					[1.0, 19*np.pi/16]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["182"] = {"32APSK 7/9" : mod}




## 32APSK - 32/45
bits_per_symbol = 5

# set the bit mapping table
R1 = 1.0/5.26
R2 = 2.84/5.26

bit_map_phasor =   [[1,	11*np.pi/16],
					[1,	9*np.pi/16],
					[1,	5*np.pi/16],
					[1,	7*np.pi/16],
					[R2, 9*np.pi/12],
					[R2, 7*np.pi/12],
					[R2, 3*np.pi/12],
					[R2, 5*np.pi/12],
					[1,	13*np.pi/16],
					[1,	15*np.pi/16],
					[1,	3*np.pi/16],
					[1,	1*np.pi/16],
					[R2, 11*np.pi/12],
					[R1, 3*np.pi/4],
					[R2, 1*np.pi/12],
					[R1, 1*np.pi/4],
					[1,	21*np.pi/16],
					[1,	23*np.pi/16],
					[1,	27*np.pi/16],
					[1,	25*np.pi/16],
					[R2, 15*np.pi/12],
					[R2, 17*np.pi/12],
					[R2, 21*np.pi/12],
					[R2, 19*np.pi/12],
					[1,	19*np.pi/16],
					[1,	17*np.pi/16],
					[1,	29*np.pi/16],
					[1,	31*np.pi/16],
					[R2, 13*np.pi/12],
					[R1, 5*np.pi/4],
					[R2, 23*np.pi/12],
					[R1, 7*np.pi/4]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["248"] = {"32APSK 32/45" : mod}




## 64APSK - 128/180
bits_per_symbol = 6

# set the bit mapping table
R1 = 1.0/3.95
R2 = 1.88/3.95
R3 = 2.72/3.95
R4 = 1.0

bit_map_phasor =   [[R1, 1*np.pi/16],
					[R1, 3*np.pi/16],
					[R1, 7*np.pi/16],
					[R1, 5*np.pi/16],
					[R1, 15*np.pi/16],
					[R1, 13*np.pi/16],
					[R1, 9*np.pi/16],
					[R1, 11*np.pi/16],
					[R1, 31*np.pi/16],
					[R1, 29*np.pi/16],
					[R1, 25*np.pi/16],
					[R1, 27*np.pi/16],
					[R1, 17*np.pi/16],
					[R1, 19*np.pi/16],
					[R1, 23*np.pi/16],
					[R1, 21*np.pi/16],
					[R2, 1*np.pi/16],
					[R2, 3*np.pi/16],
					[R2, 7*np.pi/16],
					[R2, 5*np.pi/16],
					[R2, 15*np.pi/16],
					[R2, 13*np.pi/16],
					[R2, 9*np.pi/16],
					[R2, 11*np.pi/16],
					[R2, 31*np.pi/16],
					[R2, 29*np.pi/16],
					[R2, 25*np.pi/16],
					[R2, 27*np.pi/16],
					[R2, 17*np.pi/16],
					[R2, 19*np.pi/16],
					[R2, 23*np.pi/16],
					[R2, 21*np.pi/16],
					[R4, 1*np.pi/16],
					[R4, 3*np.pi/16],
					[R4, 7*np.pi/16],
					[R4, 5*np.pi/16],
					[R4, 15*np.pi/16],
					[R4, 13*np.pi/16],
					[R4, 9*np.pi/16],
					[R4, 11*np.pi/16],
					[R4, 31*np.pi/16],
					[R4, 29*np.pi/16],
					[R4, 25*np.pi/16],
					[R4, 27*np.pi/16],
					[R4, 17*np.pi/16],
					[R4, 19*np.pi/16],
					[R4, 23*np.pi/16],
					[R4, 21*np.pi/16],
					[R3, 1*np.pi/16],
					[R3, 3*np.pi/16],
					[R3, 7*np.pi/16],
					[R3, 5*np.pi/16],
					[R3, 15*np.pi/16],
					[R3, 13*np.pi/16],
					[R3, 9*np.pi/16],
					[R3, 11*np.pi/16],
					[R3, 31*np.pi/16],
					[R3, 29*np.pi/16],
					[R3, 25*np.pi/16],
					[R3, 27*np.pi/16],
					[R3, 17*np.pi/16],
					[R3, 19*np.pi/16],
					[R3, 23*np.pi/16],
					[R3, 21*np.pi/16]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["184"] = {"64APSK 32/45-L" : mod}




## 64APSK - 132/180
bits_per_symbol = 6

# set the bit mapping table
R1 = 1.0/7.0
R2 = 2.4/7.0
R3 = 4.3/7.0
R4 = 1.0

bit_map_phasor =   [[R4, 1*np.pi/4],
					[R4, 7*np.pi/4],
					[R4, 3*np.pi/4],
					[R4, 5*np.pi/4],
					[R4, 13*np.pi/28],
					[R4, 43*np.pi/28],
					[R4, 15*np.pi/28],
					[R4, 41*np.pi/28],
					[R4, 1*np.pi/8],
					[R4, 55*np.pi/28],
					[R4, 27*np.pi/28],
					[R4, 29*np.pi/28],
					[R1, 1*np.pi/4],
					[R1, 7*np.pi/4],
					[R1, 3*np.pi/4],
					[R1, 5*np.pi/4],
					[R4, 9*np.pi/8],
					[R4, 47*np.pi/28],
					[R4, 19*np.pi/28],
					[R4, 37*np.pi/28],
					[R4, 11*np.pi/28],
					[R4, 45*np.pi/28],
					[R4, 17*np.pi/28],
					[R4, 39*np.pi/28],
					[R3, 1*np.pi/20],
					[R3, 39*np.pi/20],
					[R3, 19*np.pi/20],
					[R3, 21*np.pi/20],
					[R2, 1*np.pi/20],
					[R2, 23*np.pi/12],
					[R2, 11*np.pi/12],
					[R2, 13*np.pi/12],
					[R4, 5*np.pi/8],
					[R4, 51*np.pi/28],
					[R4, 23*np.pi/28],
					[R4, 33*np.pi/28],
					[R3, 9*np.pi/20],
					[R3, 31*np.pi/20],
					[R3, 11*np.pi/20],
					[R3, 29*np.pi/20],
					[R4, 3*np.pi/8],
					[R4, 53*np.pi/28],
					[R4, 25*np.pi/28],
					[R4, 31*np.pi/28],
					[R2, 9*np.pi/20],
					[R2, 19*np.pi/12],
					[R2, 7*np.pi/20],
					[R2, 17*np.pi/12],
					[R3, 1*np.pi/4],
					[R3, 7*np.pi/4],
					[R3, 3*np.pi/4],
					[R3, 5*np.pi/4],
					[R3, 7*np.pi/20],
					[R3, 33*np.pi/20],
					[R3, 13*np.pi/20],
					[R3, 27*np.pi/20],
					[R3, 3*np.pi/20],
					[R3, 37*np.pi/20],
					[R3, 17*np.pi/20],
					[R3, 23*np.pi/20],
					[R2, 1*np.pi/4],
					[R2, 7*np.pi/4],
					[R2, 3*np.pi/4],
					[R2, 5*np.pi/4]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["186"] = {"64APSK 11/15" : mod}




## 64APSK - 7/9, 4/5
bits_per_symbol = 6

# set the bit mapping table
R1 = 1.0/5.2
R2 = 2.2/5.2
R3 = 3.6/5.2
R4 = 1.0

bit_map_phasor =   [[R2, 25*np.pi/16],
					[R4, 7*np.pi/4],
					[R2, 27*np.pi/16],
					[R3, 7*np.pi/4],
					[R4, 31*np.pi/20],
					[R4, 33*np.pi/20],
					[R3, 31*np.pi/20],
					[R3, 33*np.pi/20],
					[R2, 23*np.pi/16],
					[R4, 5*np.pi/4],
					[R2, 21*np.pi/16],
					[R3, 5*np.pi/4],
					[R4, 29*np.pi/20],
					[R4, 27*np.pi/20],
					[R3, 29*np.pi/20],
					[R3, 27*np.pi/20],
					[R1, 13*np.pi/8],
					[R4, 37*np.pi/20],
					[R2, 29*np.pi/16],
					[R3, 37*np.pi/20],
					[R1, 15*np.pi/8],
					[R4, 39*np.pi/20],
					[R2, 31*np.pi/16],
					[R3, 39*np.pi/20],
					[R1, 11*np.pi/8],
					[R4, 23*np.pi/20],
					[R2, 19*np.pi/16],
					[R3, 23*np.pi/20],
					[R1, 9*np.pi/8],
					[R4, 21*np.pi/20],
					[R2, 17*np.pi/16],
					[R3, 21*np.pi/20],
					[R2, 7*np.pi/6],
					[R4, 1*np.pi/4],
					[R2, 5*np.pi/6],
					[R3, 1*np.pi/4],
					[R4, 9*np.pi/20],
					[R4, 7*np.pi/20],
					[R3, 9*np.pi/20],
					[R3, 7*np.pi/20],
					[R2, 9*np.pi/6],
					[R4, 3*np.pi/4],
					[R2, 11*np.pi/16],
					[R3, 3*np.pi/4],
					[R4, 11*np.pi/20],
					[R4, 13*np.pi/20],
					[R3, 11*np.pi/20],
					[R3, 13*np.pi/20],
					[R1, 3*np.pi/8],
					[R4, 3*np.pi/20],
					[R2, 3*np.pi/6],
					[R3, 3*np.pi/20],
					[R1, 1*np.pi/8],
					[R4, 1*np.pi/20],
					[R2, 1*np.pi/6],
					[R3, 1*np.pi/20],
					[R1, 5*np.pi/8],
					[R4, 17*np.pi/20],
					[R2, 13*np.pi/16],
					[R3, 17*np.pi/20],
					[R1, 7*np.pi/8],
					[R4, 19*np.pi/20],
					[R2, 15*np.pi/16],
					[R3, 19*np.pi/20]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["190"] = {"64APSK 7/9" : mod}
constellations["194"] = {"64APSK 4/5" : mod}



## 64APSK - 5/6
bits_per_symbol = 6

# set the bit mapping table
R1 = 1.0/5.0
R2 = 2.2/5.0
R3 = 3.5/5.0
R4 = 1.0

bit_map_phasor =   [[R2, 25*np.pi/16],
					[R4, 7*np.pi/4],
					[R2, 27*np.pi/16],
					[R3, 7*np.pi/4],
					[R4, 31*np.pi/20],
					[R4, 33*np.pi/20],
					[R3, 31*np.pi/20],
					[R3, 33*np.pi/20],
					[R2, 23*np.pi/16],
					[R4, 5*np.pi/4],
					[R2, 21*np.pi/16],
					[R3, 5*np.pi/4],
					[R4, 29*np.pi/20],
					[R4, 27*np.pi/20],
					[R3, 29*np.pi/20],
					[R3, 27*np.pi/20],
					[R1, 13*np.pi/8],
					[R4, 37*np.pi/20],
					[R2, 29*np.pi/16],
					[R3, 37*np.pi/20],
					[R1, 15*np.pi/8],
					[R4, 39*np.pi/20],
					[R2, 31*np.pi/16],
					[R3, 39*np.pi/20],
					[R1, 11*np.pi/8],
					[R4, 23*np.pi/20],
					[R2, 19*np.pi/16],
					[R3, 23*np.pi/20],
					[R1, 9*np.pi/8],
					[R4, 21*np.pi/20],
					[R2, 17*np.pi/16],
					[R3, 21*np.pi/20],
					[R2, 7*np.pi/6],
					[R4, 1*np.pi/4],
					[R2, 5*np.pi/6],
					[R3, 1*np.pi/4],
					[R4, 9*np.pi/20],
					[R4, 7*np.pi/20],
					[R3, 9*np.pi/20],
					[R3, 7*np.pi/20],
					[R2, 9*np.pi/6],
					[R4, 3*np.pi/4],
					[R2, 11*np.pi/16],
					[R3, 3*np.pi/4],
					[R4, 11*np.pi/20],
					[R4, 13*np.pi/20],
					[R3, 11*np.pi/20],
					[R3, 13*np.pi/20],
					[R1, 3*np.pi/8],
					[R4, 3*np.pi/20],
					[R2, 3*np.pi/6],
					[R3, 3*np.pi/20],
					[R1, 1*np.pi/8],
					[R4, 1*np.pi/20],
					[R2, 1*np.pi/6],
					[R3, 1*np.pi/20],
					[R1, 5*np.pi/8],
					[R4, 17*np.pi/20],
					[R2, 13*np.pi/16],
					[R3, 17*np.pi/20],
					[R1, 7*np.pi/8],
					[R4, 19*np.pi/20],
					[R2, 15*np.pi/16],
					[R3, 19*np.pi/20]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["198"] = {"64APSK 5/6" : mod}




## 128APSK - 135/180
bits_per_symbol = 7

# set the bit mapping table
R1 = 1.0/3.819
R2 = 1.715/3.819
R3 = 2.118/3.819
R4 = 2.681/3.819
R5 = 2.75/3.819
R6 = 1.0

bit_map_phasor =   [[R1, 83*np.pi/1260],
					[R6, 11*np.pi/105],
					[R6, 37*np.pi/1680],
					[R6, 11*np.pi/168],
					[R2, 121*np.pi/2520],
					[R3, 23*np.pi/280],
					[R5, 19*np.pi/720],
					[R4, 61*np.pi/720],
					[R1, 103*np.pi/560],
					[R6, 61*np.pi/420],
					[R6, 383*np.pi/1680],
					[R6, 929*np.pi/5040],
					[R2, 113*np.pi/560],
					[R3, 169*np.pi/1008],
					[R5, 563*np.pi/2520],
					[R4, 139*np.pi/840],
					[R1, 243*np.pi/560],
					[R6, 1993*np.pi/5040],
					[R6, 43*np.pi/90],
					[R6, 73*np.pi/168],
					[R2, 1139*np.pi/2520],
					[R3, 117*np.pi/280],
					[R5, 341*np.pi/720],
					[R4, 349*np.pi/840],
					[R1, 177*np.pi/560],
					[R6, 1789*np.pi/5040],
					[R6, 49*np.pi/180],
					[R6, 53*np.pi/168],
					[R2, 167*np.pi/560],
					[R3, 239*np.pi/720],
					[R5, 199*np.pi/720],
					[R4, 281*np.pi/840],
					[R1, 1177*np.pi/1260],
					[R6, 94*np.pi/105],
					[R6, 1643*np.pi/1680],
					[R6, 157*np.pi/168],
					[R2, 2399*np.pi/2520],
					[R3, 257*np.pi/280],
					[R5, 701*np.pi/720],
					[R4, 659*np.pi/720],
					[R1, 457*np.pi/560],
					[R6, 359*np.pi/420],
					[R6, 1297*np.pi/1680],
					[R6, 4111*np.pi/5040],
					[R2, 447*np.pi/560],
					[R3, 839*np.pi/1008],
					[R5, 1957*np.pi/2520],
					[R4, 701*np.pi/840],
					[R1, 317*np.pi/560],
					[R6, 3047*np.pi/5040],
					[R6, 47*np.pi/90],
					[R6, 95*np.pi/168],
					[R2, 1381*np.pi/2520],
					[R3, 163*np.pi/280],
					[R5, 379*np.pi/720],
					[R4, 491*np.pi/840],
					[R1, 383*np.pi/560],
					[R6, 3251*np.pi/5040],
					[R6, 131*np.pi/180],
					[R6, 115*np.pi/168],
					[R2, 393*np.pi/560],
					[R3, 481*np.pi/720],
					[R5, 521*np.pi/720],
					[R4, 559*np.pi/840],
					[R1, 2437*np.pi/1260],
					[R6, 199*np.pi/105],
					[R6, 3323*np.pi/1680],
					[R6, 325*np.pi/168],
					[R2, 4919*np.pi/2520],
					[R3, 537*np.pi/280],
					[R5, 1421*np.pi/720],
					[R4, 1379*np.pi/720],
					[R1, 1017*np.pi/560],
					[R6, 779*np.pi/420],
					[R6, 2977*np.pi/1680],
					[R6, 9151*np.pi/5040],
					[R2, 1007*np.pi/560],
					[R3, 1847*np.pi/1008],
					[R5, 4477*np.pi/2520],
					[R4, 1541*np.pi/840],
					[R1, 877*np.pi/560],
					[R6, 8087*np.pi/5040],
					[R6, 137*np.pi/90],
					[R6, 263*np.pi/168],
					[R2, 3901*np.pi/2520],
					[R3, 443*np.pi/280],
					[R5, 1099*np.pi/720],
					[R4, 1331*np.pi/840],
					[R1, 943*np.pi/560],
					[R6, 8291*np.pi/5040],
					[R6, 311*np.pi/180],
					[R6, 283*np.pi/168],
					[R2, 953*np.pi/560],
					[R3, 1201*np.pi/720],
					[R5, 1241*np.pi/720],
					[R4, 1399*np.pi/840],
					[R1, 1343*np.pi/1260],
					[R6, 116*np.pi/105],
					[R6, 1717*np.pi/1680],
					[R6, 179*np.pi/168],
					[R2, 2641*np.pi/2520],
					[R3, 303*np.pi/280],
					[R5, 739*np.pi/720],
					[R4, 781*np.pi/720],
					[R1, 663*np.pi/560],
					[R6, 481*np.pi/420],
					[R6, 2063*np.pi/1680],
					[R6, 5969*np.pi/5040],
					[R2, 673*np.pi/560],
					[R3, 1177*np.pi/1008],
					[R5, 3083*np.pi/2520],
					[R4, 979*np.pi/840],
					[R1, 803*np.pi/560],
					[R6, 7033*np.pi/5040],
					[R6, 133*np.pi/90],
					[R6, 241*np.pi/168],
					[R2, 3659*np.pi/2520],
					[R3, 397*np.pi/280],
					[R5, 1061*np.pi/720],
					[R4, 1189*np.pi/840],
					[R1, 737*np.pi/560],
					[R6, 6829*np.pi/5040],
					[R6, 229*np.pi/180],
					[R6, 221*np.pi/168],
					[R2, 727*np.pi/560],
					[R3, 959*np.pi/720],
					[R5, 919*np.pi/720],
					[R4, 1121*np.pi/840]]


# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["200"] = {"128APSK 3/4" : mod}



## 128APSK - 140/180
bits_per_symbol = 7

# set the bit mapping table
R1 = 1.0/3.733
R2 = 1.715/3.733
R3 = 2.118/3.733
R4 = 2.681/3.733
R5 = 2.75/3.733
R6 = 1.0

bit_map_phasor =   [[R1, 83*np.pi/1260],
					[R6, 11*np.pi/105],
					[R6, 37*np.pi/1680],
					[R6, 11*np.pi/168],
					[R2, 121*np.pi/2520],
					[R3, 23*np.pi/280],
					[R5, 19*np.pi/720],
					[R4, 61*np.pi/720],
					[R1, 103*np.pi/560],
					[R6, 61*np.pi/420],
					[R6, 383*np.pi/1680],
					[R6, 929*np.pi/5040],
					[R2, 113*np.pi/560],
					[R3, 169*np.pi/1008],
					[R5, 563*np.pi/2520],
					[R4, 139*np.pi/840],
					[R1, 243*np.pi/560],
					[R6, 1993*np.pi/5040],
					[R6, 43*np.pi/90],
					[R6, 73*np.pi/168],
					[R2, 1139*np.pi/2520],
					[R3, 117*np.pi/280],
					[R5, 341*np.pi/720],
					[R4, 349*np.pi/840],
					[R1, 177*np.pi/560],
					[R6, 1789*np.pi/5040],
					[R6, 49*np.pi/180],
					[R6, 53*np.pi/168],
					[R2, 167*np.pi/560],
					[R3, 239*np.pi/720],
					[R5, 199*np.pi/720],
					[R4, 281*np.pi/840],
					[R1, 1177*np.pi/1260],
					[R6, 94*np.pi/105],
					[R6, 1643*np.pi/1680],
					[R6, 157*np.pi/168],
					[R2, 2399*np.pi/2520],
					[R3, 257*np.pi/280],
					[R5, 701*np.pi/720],
					[R4, 659*np.pi/720],
					[R1, 457*np.pi/560],
					[R6, 359*np.pi/420],
					[R6, 1297*np.pi/1680],
					[R6, 4111*np.pi/5040],
					[R2, 447*np.pi/560],
					[R3, 839*np.pi/1008],
					[R5, 1957*np.pi/2520],
					[R4, 701*np.pi/840],
					[R1, 317*np.pi/560],
					[R6, 3047*np.pi/5040],
					[R6, 47*np.pi/90],
					[R6, 95*np.pi/168],
					[R2, 1381*np.pi/2520],
					[R3, 163*np.pi/280],
					[R5, 379*np.pi/720],
					[R4, 491*np.pi/840],
					[R1, 383*np.pi/560],
					[R6, 3251*np.pi/5040],
					[R6, 131*np.pi/180],
					[R6, 115*np.pi/168],
					[R2, 393*np.pi/560],
					[R3, 481*np.pi/720],
					[R5, 521*np.pi/720],
					[R4, 559*np.pi/840],
					[R1, 2437*np.pi/1260],
					[R6, 199*np.pi/105],
					[R6, 3323*np.pi/1680],
					[R6, 325*np.pi/168],
					[R2, 4919*np.pi/2520],
					[R3, 537*np.pi/280],
					[R5, 1421*np.pi/720],
					[R4, 1379*np.pi/720],
					[R1, 1017*np.pi/560],
					[R6, 779*np.pi/420],
					[R6, 2977*np.pi/1680],
					[R6, 9151*np.pi/5040],
					[R2, 1007*np.pi/560],
					[R3, 1847*np.pi/1008],
					[R5, 4477*np.pi/2520],
					[R4, 1541*np.pi/840],
					[R1, 877*np.pi/560],
					[R6, 8087*np.pi/5040],
					[R6, 137*np.pi/90],
					[R6, 263*np.pi/168],
					[R2, 3901*np.pi/2520],
					[R3, 443*np.pi/280],
					[R5, 1099*np.pi/720],
					[R4, 1331*np.pi/840],
					[R1, 943*np.pi/560],
					[R6, 8291*np.pi/5040],
					[R6, 311*np.pi/180],
					[R6, 283*np.pi/168],
					[R2, 953*np.pi/560],
					[R3, 1201*np.pi/720],
					[R5, 1241*np.pi/720],
					[R4, 1399*np.pi/840],
					[R1, 1343*np.pi/1260],
					[R6, 116*np.pi/105],
					[R6, 1717*np.pi/1680],
					[R6, 179*np.pi/168],
					[R2, 2641*np.pi/2520],
					[R3, 303*np.pi/280],
					[R5, 739*np.pi/720],
					[R4, 781*np.pi/720],
					[R1, 663*np.pi/560],
					[R6, 481*np.pi/420],
					[R6, 2063*np.pi/1680],
					[R6, 5969*np.pi/5040],
					[R2, 673*np.pi/560],
					[R3, 1177*np.pi/1008],
					[R5, 3083*np.pi/2520],
					[R4, 979*np.pi/840],
					[R1, 803*np.pi/560],
					[R6, 7033*np.pi/5040],
					[R6, 133*np.pi/90],
					[R6, 241*np.pi/168],
					[R2, 3659*np.pi/2520],
					[R3, 397*np.pi/280],
					[R5, 1061*np.pi/720],
					[R4, 1189*np.pi/840],
					[R1, 737*np.pi/560],
					[R6, 6829*np.pi/5040],
					[R6, 229*np.pi/180],
					[R6, 221*np.pi/168],
					[R2, 727*np.pi/560],
					[R3, 959*np.pi/720],
					[R5, 919*np.pi/720],
					[R4, 1121*np.pi/840]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["202"] = {"128APSK 7/9" : mod}





## 256APSK - 116/180, 124/180
bits_per_symbol = 8

# set the bit mapping table
R1 = 1.0/6.536
R2 = 1.791/6.536
R3 = 2.405/6.536
R4 = 2.980/6.536
R5 = 3.569/6.536
R6 = 4.235/6.536
R7 = 5.078/6.536
R8 = 1.0

bit_map_phasor =   [[R1, 1*np.pi/32],
					[R1, 3*np.pi/32],
					[R1, 7*np.pi/32],
					[R1, 5*np.pi/32],
					[R1, 15*np.pi/32],
					[R1, 13*np.pi/32],
					[R1, 9*np.pi/32],
					[R1, 11*np.pi/32],
					[R1, 31*np.pi/32],
					[R1, 29*np.pi/32],
					[R1, 25*np.pi/32],
					[R1, 27*np.pi/32],
					[R1, 17*np.pi/32],
					[R1, 19*np.pi/32],
					[R1, 23*np.pi/32],
					[R1, 21*np.pi/32],
					[R1, 63*np.pi/32],
					[R1, 61*np.pi/32],
					[R1, 57*np.pi/32],
					[R1, 59*np.pi/32],
					[R1, 49*np.pi/32],
					[R1, 51*np.pi/32],
					[R1, 55*np.pi/32],
					[R1, 53*np.pi/32],
					[R1, 33*np.pi/32],
					[R1, 35*np.pi/32],
					[R1, 39*np.pi/32],
					[R1, 37*np.pi/32],
					[R1, 47*np.pi/32],
					[R1, 45*np.pi/32],
					[R1, 41*np.pi/32],
					[R1, 43*np.pi/32],
					[R2, 1*np.pi/32],
					[R2, 3*np.pi/32],
					[R2, 7*np.pi/32],
					[R2, 5*np.pi/32],
					[R2, 15*np.pi/32],
					[R2, 13*np.pi/32],
					[R2, 9*np.pi/32],
					[R2, 11*np.pi/32],
					[R2, 31*np.pi/32],
					[R2, 29*np.pi/32],
					[R2, 25*np.pi/32],
					[R2, 27*np.pi/32],
					[R2, 17*np.pi/32],
					[R2, 19*np.pi/32],
					[R2, 23*np.pi/32],
					[R2, 21*np.pi/32],
					[R2, 63*np.pi/32],
					[R2, 61*np.pi/32],
					[R2, 57*np.pi/32],
					[R2, 59*np.pi/32],
					[R2, 49*np.pi/32],
					[R2, 51*np.pi/32],
					[R2, 55*np.pi/32],
					[R2, 53*np.pi/32],
					[R2, 33*np.pi/32],
					[R2, 35*np.pi/32],
					[R2, 39*np.pi/32],
					[R2, 37*np.pi/32],
					[R2, 47*np.pi/32],
					[R2, 45*np.pi/32],
					[R2, 41*np.pi/32],
					[R2, 43*np.pi/32],
					[R4, 1*np.pi/32],
					[R4, 3*np.pi/32],
					[R4, 7*np.pi/32],
					[R4, 5*np.pi/32],
					[R4, 15*np.pi/32],
					[R4, 13*np.pi/32],
					[R4, 9*np.pi/32],
					[R4, 11*np.pi/32],
					[R4, 31*np.pi/32],
					[R4, 29*np.pi/32],
					[R4, 25*np.pi/32],
					[R4, 27*np.pi/32],
					[R4, 17*np.pi/32],
					[R4, 19*np.pi/32],
					[R4, 23*np.pi/32],
					[R4, 21*np.pi/32],
					[R4, 63*np.pi/32],
					[R4, 61*np.pi/32],
					[R4, 57*np.pi/32],
					[R4, 59*np.pi/32],
					[R4, 49*np.pi/32],
					[R4, 51*np.pi/32],
					[R4, 55*np.pi/32],
					[R4, 53*np.pi/32],
					[R4, 33*np.pi/32],
					[R4, 35*np.pi/32],
					[R4, 39*np.pi/32],
					[R4, 37*np.pi/32],
					[R4, 47*np.pi/32],
					[R4, 45*np.pi/32],
					[R4, 41*np.pi/32],
					[R4, 43*np.pi/32],
					[R3, 1*np.pi/32],
					[R3, 3*np.pi/32],
					[R3, 7*np.pi/32],
					[R3, 5*np.pi/32],
					[R3, 15*np.pi/32],
					[R3, 13*np.pi/32],
					[R3, 9*np.pi/32],
					[R3, 11*np.pi/32],
					[R3, 31*np.pi/32],
					[R3, 29*np.pi/32],
					[R3, 25*np.pi/32],
					[R3, 27*np.pi/32],
					[R3, 17*np.pi/32],
					[R3, 19*np.pi/32],
					[R3, 23*np.pi/32],
					[R3, 21*np.pi/32],
					[R3, 63*np.pi/32],
					[R3, 61*np.pi/32],
					[R3, 57*np.pi/32],
					[R3, 59*np.pi/32],
					[R3, 49*np.pi/32],
					[R3, 51*np.pi/32],
					[R3, 55*np.pi/32],
					[R3, 53*np.pi/32],
					[R3, 33*np.pi/32],
					[R3, 35*np.pi/32],
					[R3, 39*np.pi/32],
					[R3, 37*np.pi/32],
					[R3, 47*np.pi/32],
					[R3, 45*np.pi/32],
					[R3, 41*np.pi/32],
					[R3, 43*np.pi/32],
					[R8, 1*np.pi/32],
					[R8, 3*np.pi/32],
					[R8, 7*np.pi/32],
					[R8, 5*np.pi/32],
					[R8, 15*np.pi/32],
					[R8, 13*np.pi/32],
					[R8, 9*np.pi/32],
					[R8, 11*np.pi/32],
					[R8, 31*np.pi/32],
					[R8, 29*np.pi/32],
					[R8, 25*np.pi/32],
					[R8, 27*np.pi/32],
					[R8, 17*np.pi/32],
					[R8, 19*np.pi/32],
					[R8, 23*np.pi/32],
					[R8, 21*np.pi/32],
					[R8, 63*np.pi/32],
					[R8, 61*np.pi/32],
					[R8, 57*np.pi/32],
					[R8, 59*np.pi/32],
					[R8, 49*np.pi/32],
					[R8, 51*np.pi/32],
					[R8, 55*np.pi/32],
					[R8, 53*np.pi/32],
					[R8, 33*np.pi/32],
					[R8, 35*np.pi/32],
					[R8, 39*np.pi/32],
					[R8, 37*np.pi/32],
					[R8, 47*np.pi/32],
					[R8, 45*np.pi/32],
					[R8, 41*np.pi/32],
					[R8, 43*np.pi/32],
					[R7, 1*np.pi/32],
					[R7, 3*np.pi/32],
					[R7, 7*np.pi/32],
					[R7, 5*np.pi/32],
					[R7, 15*np.pi/32],
					[R7, 13*np.pi/32],
					[R7, 9*np.pi/32],
					[R7, 11*np.pi/32],
					[R7, 31*np.pi/32],
					[R7, 29*np.pi/32],
					[R7, 25*np.pi/32],
					[R7, 27*np.pi/32],
					[R7, 17*np.pi/32],
					[R7, 19*np.pi/32],
					[R7, 23*np.pi/32],
					[R7, 21*np.pi/32],
					[R7, 63*np.pi/32],
					[R7, 61*np.pi/32],
					[R7, 57*np.pi/32],
					[R7, 59*np.pi/32],
					[R7, 49*np.pi/32],
					[R7, 51*np.pi/32],
					[R7, 55*np.pi/32],
					[R7, 53*np.pi/32],
					[R7, 33*np.pi/32],
					[R7, 35*np.pi/32],
					[R7, 39*np.pi/32],
					[R7, 37*np.pi/32],
					[R7, 47*np.pi/32],
					[R7, 45*np.pi/32],
					[R7, 41*np.pi/32],
					[R7, 43*np.pi/32],
					[R5, 1*np.pi/32],
					[R5, 3*np.pi/32],
					[R5, 7*np.pi/32],
					[R5, 5*np.pi/32],
					[R5, 15*np.pi/32],
					[R5, 13*np.pi/32],
					[R5, 9*np.pi/32],
					[R5, 11*np.pi/32],
					[R5, 31*np.pi/32],
					[R5, 29*np.pi/32],
					[R5, 25*np.pi/32],
					[R5, 27*np.pi/32],
					[R5, 17*np.pi/32],
					[R5, 19*np.pi/32],
					[R5, 23*np.pi/32],
					[R5, 21*np.pi/32],
					[R5, 63*np.pi/32],
					[R5, 61*np.pi/32],
					[R5, 57*np.pi/32],
					[R5, 59*np.pi/32],
					[R5, 49*np.pi/32],
					[R5, 51*np.pi/32],
					[R5, 55*np.pi/32],
					[R5, 53*np.pi/32],
					[R5, 33*np.pi/32],
					[R5, 35*np.pi/32],
					[R5, 39*np.pi/32],
					[R5, 37*np.pi/32],
					[R5, 47*np.pi/32],
					[R5, 45*np.pi/32],
					[R5, 41*np.pi/32],
					[R5, 43*np.pi/32],
					[R6, 1*np.pi/32],
					[R6, 3*np.pi/32],
					[R6, 7*np.pi/32],
					[R6, 5*np.pi/32],
					[R6, 15*np.pi/32],
					[R6, 13*np.pi/32],
					[R6, 9*np.pi/32],
					[R6, 11*np.pi/32],
					[R6, 31*np.pi/32],
					[R6, 29*np.pi/32],
					[R6, 25*np.pi/32],
					[R6, 27*np.pi/32],
					[R6, 17*np.pi/32],
					[R6, 19*np.pi/32],
					[R6, 23*np.pi/32],
					[R6, 21*np.pi/32],
					[R6, 63*np.pi/32],
					[R6, 61*np.pi/32],
					[R6, 57*np.pi/32],
					[R6, 59*np.pi/32],
					[R6, 49*np.pi/32],
					[R6, 51*np.pi/32],
					[R6, 55*np.pi/32],
					[R6, 53*np.pi/32],
					[R6, 33*np.pi/32],
					[R6, 35*np.pi/32],
					[R6, 39*np.pi/32],
					[R6, 37*np.pi/32],
					[R6, 47*np.pi/32],
					[R6, 45*np.pi/32],
					[R6, 41*np.pi/32],
					[R6, 43*np.pi/32]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["204"] = {"256APSK 29/45-L" : mod}
constellations["208"] = {"256APSK 31/45-L" : mod}



## 256APSK - 20/30
bits_per_symbol = 8

# set the bit mapping table
R1 = 1.0/6.536
R2 = 1.791/6.536
R3 = 2.405/6.536
R4 = 2.980/6.536
R5 = 3.569/6.536
R6 = 4.235/6.536
R7 = 5.078/6.536
R8 = 1.0

bit_map_cartesian =[[1.635,		0.1593],
					[1.5776,	0.4735],
					[0.943,		0.11],
					[0.9069,	0.2829],
					[0.3237,	0.0849],
					[0.3228,	0.0867],
					[0.7502,	0.1138],
					[0.7325,	0.2088],
					[0.1658,	1.6747],
					[0.4907,	1.6084],
					[0.1088,	0.953],
					[0.2464,	0.927],
					[0.0872,	0.139],
					[0.0871,	0.1392],
					[0.1091,	0.7656],
					[0.1699,	0.7537],
					[-1.635,	0.1593],
					[-1.5776,	0.4735],
					[-0.943,	0.11],
					[-0.9069,	0.2829],
					[-0.3237,	0.0849],
					[-0.3228,	0.0867],
					[-0.7502,	0.1138],
					[-0.7325,	0.2088],
					[-0.1658,	1.6747],
					[-0.4907,	1.6084],
					[-0.1088,	0.953],
					[-0.2464,	0.927],
					[-0.0872,	0.139],
					[-0.0871,	0.1392],
					[-0.1091,	0.7656],
					[-0.1699,	0.7537],
					[1.3225,	0.132],
					[1.2742,	0.3922],
					[1.0854,	0.1139],
					[1.0441,	0.3296],
					[0.4582,	0.1123],
					[0.4545,	0.1251],
					[0.6473,	0.1138],
					[0.6339,	0.1702],
					[0.1322,	1.3631],
					[0.3929,	1.3102],
					[0.1124,	1.1327],
					[0.316,	1.0913],
					[0.0928,	0.397],
					[0.0937,	0.3973],
					[0.1054,	0.5979],
					[0.123,	0.5949],
					[-1.3225,	0.132],
					[-1.2742,	0.3922],
					[-1.0854,	0.1139],
					[-1.0441,	0.3296],
					[-0.4582,	0.1123],
					[-0.4545,	0.1251],
					[-0.6473,	0.1138],
					[-0.6339,	0.1702],
					[-0.1322,	1.3631],
					[-0.3929,	1.3102],
					[-0.1124,	1.1327],
					[-0.316,	1.0913],
					[-0.0928,	0.397],
					[-0.0937,	0.3973],
					[-0.1054,	0.5979],
					[-0.123,	0.5949],
					[1.635,	0.1593],
					[1.5776,	0.4735],
					[0.943,	0.11],
					[0.9069,	0.2829],
					[0.3237,	0.0849],
					[0.3228,	0.0867],
					[0.7502,	0.1138],
					[0.7325,	0.2088],
					[0.1658,	1.6747],
					[0.4907,	1.6084],
					[0.1088,	0.953],
					[0.2464,	0.927],
					[0.0872,	0.139],
					[0.0871,	0.1392],
					[0.1091,	0.7656],
					[0.1699,	0.7537],
					[-1.635,	0.1593],
					[-1.5776,	0.4735],
					[-0.943,	0.11],
					[-0.9069,	0.2829],
					[-0.3237,	0.0849],
					[-0.3228,	0.0867],
					[-0.7502,	0.1138],
					[-0.7325,	0.2088],
					[-0.1658,	1.6747],
					[-0.4907,	1.6084],
					[-0.1088,	0.953],
					[-0.2464,	0.927],
					[-0.0872,	0.139],
					[-0.0871,	0.1392],
					[-0.1091,	0.7656],
					[-0.1699,	0.7537],
					[1.3225,	0.132],
					[1.2742,	0.3922],
					[1.0854,	0.1139],
					[1.0441,	0.3296],
					[0.4582,	0.1123],
					[0.4545,	0.1251],
					[0.6473,	0.1138],
					[0.6339,	0.1702],
					[0.1322,	1.3631],
					[0.3929,	1.3102],
					[0.1124,	1.1327],
					[0.316,	1.0913],
					[0.0928,	0.397],
					[0.0937,	0.3973],
					[0.1054,	0.5979],
					[0.123,	0.5949],
					[-1.3225,	0.132],
					[-1.2742,	0.3922],
					[-1.0854,	0.1139],
					[-1.0441,	0.3296],
					[-0.4582,	0.1123],
					[-0.4545,	0.1251],
					[-0.6473,	0.1138],
					[-0.6339,	0.1702],
					[-0.1322,	1.3631],
					[-0.3929,	1.3102],
					[-0.1124,	1.1327],
					[-0.316,	1.0913],
					[-0.0928,	0.397],
					[-0.0937,	0.3973],
					[-0.1054,	0.5979],
					[-0.123,	0.5949],
					[1.2901,	1.0495],
					[1.4625,	0.774],
					[0.7273,	0.616],
					[0.8177,	0.4841],
					[0.2844,	0.1296],
					[0.2853,	0.1309],
					[0.5902,	0.4857],
					[0.6355,	0.4185],
					[1.0646,	1.2876],
					[0.7949,	1.4772],
					[0.5707,	0.7662],
					[0.449,	0.8461],
					[0.1053,	0.1494],
					[0.1052,	0.1495],
					[0.4294,	0.6363],
					[0.3744,	0.6744],
					[-1.2901,	1.0495],
					[-1.4625,	0.774],
					[-0.7273,	0.616],
					[-0.8177,	0.4841],
					[-0.2844,	0.1296],
					[-0.2853,	0.1309],
					[-0.5902,	0.4857],
					[-0.6355,	0.4185],
					[-1.0646,	1.2876],
					[-0.7949,	1.4772],
					[-0.5707,	0.7662],
					[-0.449,	0.8461],
					[-0.1053,	0.1494],
					[-0.1052,	0.1495],
					[-0.4294,	0.6363],
					[-0.3744,	0.6744],
					[1.0382,	0.8623],
					[1.1794,	0.6376],
					[0.8504,	0.7217],
					[0.9638,	0.5407],
					[0.3734,	0.256],
					[0.3799,	0.2517],
					[0.4968,	0.3947],
					[0.5231,	0.3644],
					[0.8555,	1.0542],
					[0.6363,	1.2064],
					[0.6961,	0.885],
					[0.5229,	1.0037],
					[0.1938,	0.3621],
					[0.1909,	0.3627],
					[0.3224,	0.5236],
					[0.3016,	0.5347],
					[-1.0382,	0.8623],
					[-1.1794,	0.6376],
					[-0.8504,	0.7217],
					[-0.9638,	0.5407],
					[-0.3734,	0.256],
					[-0.3799,	0.2517],
					[-0.4968,	0.3947],
					[-0.5231,	0.3644],
					[-0.8555,	1.0542],
					[-0.6363,	1.2064],
					[-0.6961,	0.885],
					[-0.5229,	1.0037],
					[-0.1938,	0.3621],
					[-0.1909,	0.3627],
					[-0.3224,	0.5236],
					[-0.3016,	0.5347],
					[1.2901,	1.0495],
					[1.4625,	0.774],
					[0.7273,	0.616],
					[0.8177,	0.4841],
					[0.2844,	0.1296],
					[0.2853,	0.1309],
					[0.5902,	0.4857],
					[0.6355,	0.4185],
					[1.0646,	1.2876],
					[0.7949,	1.4772],
					[0.5707,	0.7662],
					[0.449,	0.8461],
					[0.1053,	0.1494],
					[0.1052,	0.1495],
					[0.4294,	0.6363],
					[0.3744,	0.6744],
					[-1.2901,	1.0495],
					[-1.4625,	0.774],
					[-0.7273,	0.616],
					[-0.8177,	0.4841],
					[-0.2844,	0.1296],
					[-0.2853,	0.1309],
					[-0.5902,	0.4857],
					[-0.6355,	0.4185],
					[-1.0646,	1.2876],
					[-0.7949,	1.4772],
					[-0.5707,	0.7662],
					[-0.449,	0.8461],
					[-0.1053,	0.1494],
					[-0.1052,	0.1495],
					[-0.4294,	0.6363],
					[-0.3744,	0.6744],
					[1.0382,	0.8623],
					[1.1794,	0.6376],
					[0.8504,	0.7217],
					[0.9638,	0.5407],
					[0.3734,	0.256],
					[0.3799,	0.2517],
					[0.4968,	0.3947],
					[0.5231,	0.3644],
					[0.8555,	1.0542],
					[0.6363,	1.2064],
					[0.6961,	0.885],
					[0.5229,	1.0037],
					[0.1938,	0.3621],
					[0.1909,	0.3627],
					[0.3224,	0.5236],
					[0.3016,	0.5347],
					[-1.0382,	0.8623],
					[-1.1794,	0.6376],
					[-0.8504,	0.7217],
					[-0.9638,	0.5407],
					[-0.3734,	0.256],
					[-0.3799,	0.2517],
					[-0.4968,	0.3947],
					[-0.5231,	0.3644],
					[-0.8555,	1.0542],
					[-0.6363,	1.2064],
					[-0.6961,	0.885],
					[-0.5229,	1.0037],
					[-0.1938,	0.3621],
					[-0.1909,	0.3627],
					[-0.3224,	0.5236],
					[-0.3016,	0.5347]]

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["206"] = {"256APSK 2/3-L" : mod}



## 256APSK - 128/180
bits_per_symbol = 8

# set the bit mapping table
R1 = 1.0/5.4
R2 = 1.794/5.4
R3 = 2.409/5.4
R4 = 2.986/5.4
R5 = 3.579/5.4
R6 = 4.045/5.4
R7 = 4.6/5.4
R8 = 1.0

bit_map_phasor =   [[R1, 1*np.pi/32],
					[R1, 3*np.pi/32],
					[R1, 7*np.pi/32],
					[R1, 5*np.pi/32],
					[R1, 15*np.pi/32],
					[R1, 13*np.pi/32],
					[R1, 9*np.pi/32],
					[R1, 11*np.pi/32],
					[R1, 31*np.pi/32],
					[R1, 29*np.pi/32],
					[R1, 25*np.pi/32],
					[R1, 27*np.pi/32],
					[R1, 17*np.pi/32],
					[R1, 19*np.pi/32],
					[R1, 23*np.pi/32],
					[R1, 21*np.pi/32],
					[R1, 63*np.pi/32],
					[R1, 61*np.pi/32],
					[R1, 57*np.pi/32],
					[R1, 59*np.pi/32],
					[R1, 49*np.pi/32],
					[R1, 51*np.pi/32],
					[R1, 55*np.pi/32],
					[R1, 53*np.pi/32],
					[R1, 33*np.pi/32],
					[R1, 35*np.pi/32],
					[R1, 39*np.pi/32],
					[R1, 37*np.pi/32],
					[R1, 47*np.pi/32],
					[R1, 45*np.pi/32],
					[R1, 41*np.pi/32],
					[R1, 43*np.pi/32],
					[R2, 1*np.pi/32],
					[R2, 3*np.pi/32],
					[R2, 7*np.pi/32],
					[R2, 5*np.pi/32],
					[R2, 15*np.pi/32],
					[R2, 13*np.pi/32],
					[R2, 9*np.pi/32],
					[R2, 11*np.pi/32],
					[R2, 31*np.pi/32],
					[R2, 29*np.pi/32],
					[R2, 25*np.pi/32],
					[R2, 27*np.pi/32],
					[R2, 17*np.pi/32],
					[R2, 19*np.pi/32],
					[R2, 23*np.pi/32],
					[R2, 21*np.pi/32],
					[R2, 63*np.pi/32],
					[R2, 61*np.pi/32],
					[R2, 57*np.pi/32],
					[R2, 59*np.pi/32],
					[R2, 49*np.pi/32],
					[R2, 51*np.pi/32],
					[R2, 55*np.pi/32],
					[R2, 53*np.pi/32],
					[R2, 33*np.pi/32],
					[R2, 35*np.pi/32],
					[R2, 39*np.pi/32],
					[R2, 37*np.pi/32],
					[R2, 47*np.pi/32],
					[R2, 45*np.pi/32],
					[R2, 41*np.pi/32],
					[R2, 43*np.pi/32],
					[R4, 1*np.pi/32],
					[R4, 3*np.pi/32],
					[R4, 7*np.pi/32],
					[R4, 5*np.pi/32],
					[R4, 15*np.pi/32],
					[R4, 13*np.pi/32],
					[R4, 9*np.pi/32],
					[R4, 11*np.pi/32],
					[R4, 31*np.pi/32],
					[R4, 29*np.pi/32],
					[R4, 25*np.pi/32],
					[R4, 27*np.pi/32],
					[R4, 17*np.pi/32],
					[R4, 19*np.pi/32],
					[R4, 23*np.pi/32],
					[R4, 21*np.pi/32],
					[R4, 63*np.pi/32],
					[R4, 61*np.pi/32],
					[R4, 57*np.pi/32],
					[R4, 59*np.pi/32],
					[R4, 49*np.pi/32],
					[R4, 51*np.pi/32],
					[R4, 55*np.pi/32],
					[R4, 53*np.pi/32],
					[R4, 33*np.pi/32],
					[R4, 35*np.pi/32],
					[R4, 39*np.pi/32],
					[R4, 37*np.pi/32],
					[R4, 47*np.pi/32],
					[R4, 45*np.pi/32],
					[R4, 41*np.pi/32],
					[R4, 43*np.pi/32],
					[R3, 1*np.pi/32],
					[R3, 3*np.pi/32],
					[R3, 7*np.pi/32],
					[R3, 5*np.pi/32],
					[R3, 15*np.pi/32],
					[R3, 13*np.pi/32],
					[R3, 9*np.pi/32],
					[R3, 11*np.pi/32],
					[R3, 31*np.pi/32],
					[R3, 29*np.pi/32],
					[R3, 25*np.pi/32],
					[R3, 27*np.pi/32],
					[R3, 17*np.pi/32],
					[R3, 19*np.pi/32],
					[R3, 23*np.pi/32],
					[R3, 21*np.pi/32],
					[R3, 63*np.pi/32],
					[R3, 61*np.pi/32],
					[R3, 57*np.pi/32],
					[R3, 59*np.pi/32],
					[R3, 49*np.pi/32],
					[R3, 51*np.pi/32],
					[R3, 55*np.pi/32],
					[R3, 53*np.pi/32],
					[R3, 33*np.pi/32],
					[R3, 35*np.pi/32],
					[R3, 39*np.pi/32],
					[R3, 37*np.pi/32],
					[R3, 47*np.pi/32],
					[R3, 45*np.pi/32],
					[R3, 41*np.pi/32],
					[R3, 43*np.pi/32],
					[R8, 1*np.pi/32],
					[R8, 3*np.pi/32],
					[R8, 7*np.pi/32],
					[R8, 5*np.pi/32],
					[R8, 15*np.pi/32],
					[R8, 13*np.pi/32],
					[R8, 9*np.pi/32],
					[R8, 11*np.pi/32],
					[R8, 31*np.pi/32],
					[R8, 29*np.pi/32],
					[R8, 25*np.pi/32],
					[R8, 27*np.pi/32],
					[R8, 17*np.pi/32],
					[R8, 19*np.pi/32],
					[R8, 23*np.pi/32],
					[R8, 21*np.pi/32],
					[R8, 63*np.pi/32],
					[R8, 61*np.pi/32],
					[R8, 57*np.pi/32],
					[R8, 59*np.pi/32],
					[R8, 49*np.pi/32],
					[R8, 51*np.pi/32],
					[R8, 55*np.pi/32],
					[R8, 53*np.pi/32],
					[R8, 33*np.pi/32],
					[R8, 35*np.pi/32],
					[R8, 39*np.pi/32],
					[R8, 37*np.pi/32],
					[R8, 47*np.pi/32],
					[R8, 45*np.pi/32],
					[R8, 41*np.pi/32],
					[R8, 43*np.pi/32],
					[R7, 1*np.pi/32],
					[R7, 3*np.pi/32],
					[R7, 7*np.pi/32],
					[R7, 5*np.pi/32],
					[R7, 15*np.pi/32],
					[R7, 13*np.pi/32],
					[R7, 9*np.pi/32],
					[R7, 11*np.pi/32],
					[R7, 31*np.pi/32],
					[R7, 29*np.pi/32],
					[R7, 25*np.pi/32],
					[R7, 27*np.pi/32],
					[R7, 17*np.pi/32],
					[R7, 19*np.pi/32],
					[R7, 23*np.pi/32],
					[R7, 21*np.pi/32],
					[R7, 63*np.pi/32],
					[R7, 61*np.pi/32],
					[R7, 57*np.pi/32],
					[R7, 59*np.pi/32],
					[R7, 49*np.pi/32],
					[R7, 51*np.pi/32],
					[R7, 55*np.pi/32],
					[R7, 53*np.pi/32],
					[R7, 33*np.pi/32],
					[R7, 35*np.pi/32],
					[R7, 39*np.pi/32],
					[R7, 37*np.pi/32],
					[R7, 47*np.pi/32],
					[R7, 45*np.pi/32],
					[R7, 41*np.pi/32],
					[R7, 43*np.pi/32],
					[R5, 1*np.pi/32],
					[R5, 3*np.pi/32],
					[R5, 7*np.pi/32],
					[R5, 5*np.pi/32],
					[R5, 15*np.pi/32],
					[R5, 13*np.pi/32],
					[R5, 9*np.pi/32],
					[R5, 11*np.pi/32],
					[R5, 31*np.pi/32],
					[R5, 29*np.pi/32],
					[R5, 25*np.pi/32],
					[R5, 27*np.pi/32],
					[R5, 17*np.pi/32],
					[R5, 19*np.pi/32],
					[R5, 23*np.pi/32],
					[R5, 21*np.pi/32],
					[R5, 63*np.pi/32],
					[R5, 61*np.pi/32],
					[R5, 57*np.pi/32],
					[R5, 59*np.pi/32],
					[R5, 49*np.pi/32],
					[R5, 51*np.pi/32],
					[R5, 55*np.pi/32],
					[R5, 53*np.pi/32],
					[R5, 33*np.pi/32],
					[R5, 35*np.pi/32],
					[R5, 39*np.pi/32],
					[R5, 37*np.pi/32],
					[R5, 47*np.pi/32],
					[R5, 45*np.pi/32],
					[R5, 41*np.pi/32],
					[R5, 43*np.pi/32],
					[R6, 1*np.pi/32],
					[R6, 3*np.pi/32],
					[R6, 7*np.pi/32],
					[R6, 5*np.pi/32],
					[R6, 15*np.pi/32],
					[R6, 13*np.pi/32],
					[R6, 9*np.pi/32],
					[R6, 11*np.pi/32],
					[R6, 31*np.pi/32],
					[R6, 29*np.pi/32],
					[R6, 25*np.pi/32],
					[R6, 27*np.pi/32],
					[R6, 17*np.pi/32],
					[R6, 19*np.pi/32],
					[R6, 23*np.pi/32],
					[R6, 21*np.pi/32],
					[R6, 63*np.pi/32],
					[R6, 61*np.pi/32],
					[R6, 57*np.pi/32],
					[R6, 59*np.pi/32],
					[R6, 49*np.pi/32],
					[R6, 51*np.pi/32],
					[R6, 55*np.pi/32],
					[R6, 53*np.pi/32],
					[R6, 33*np.pi/32],
					[R6, 35*np.pi/32],
					[R6, 39*np.pi/32],
					[R6, 37*np.pi/32],
					[R6, 47*np.pi/32],
					[R6, 45*np.pi/32],
					[R6, 41*np.pi/32],
					[R6, 43*np.pi/32]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 

# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["210"] = {"256APSK 32/45" : mod}



## 256APSK - 22/30
bits_per_symbol = 8

bit_map_cartesian =[[1.5977,	0.1526],
					[1.3187,	0.1269],
					[-1.5977,	0.1526],
					[-1.3187,	0.1269],
					[0.2574,	0.0733],
					[0.4496,	0.0807],
					[-0.2574,	0.0733],
					[-0.4496,	0.0807],
					[1.5977,	0.1526],
					[1.3187,	0.1269],
					[-1.5977,	0.1526],
					[-1.3187,	0.1269],
					[0.2574,	0.0733],
					[0.4496,	0.0807],
					[-0.2574,	0.0733],
					[-0.4496,	0.0807],
					[0.9269,	0.0943],
					[1.1024,	0.1086],
					[-0.9269,	0.0943],
					[-1.1024,	0.1086],
					[0.7663,	0.0867],
					[0.6115,	0.0871],
					[-0.7663,	0.0867],
					[-0.6115,	0.0871],
					[0.9269,	0.0943],
					[1.1024,	0.1086],
					[-0.9269,	0.0943],
					[-1.1024,	0.1086],
					[0.7663,	0.0867],
					[0.6115,	0.0871],
					[-0.7663,	0.0867],
					[-0.6115,	0.0871],
					[1.2701,	1.0139],
					[1.0525,	0.8406],
					[-1.2701,	1.0139],
					[-1.0525,	0.8406],
					[0.2487,	0.1978],
					[0.3523,	0.2915],
					[-0.2487,	0.1978],
					[-0.3523,	0.2915],
					[1.2701,	1.0139],
					[1.0525,	0.8406],
					[-1.2701,	1.0139],
					[-1.0525,	0.8406],
					[0.2487,	0.1978],
					[0.3523,	0.2915],
					[-0.2487,	0.1978],
					[-0.3523,	0.2915],
					[0.7359,	0.6043],
					[0.8807,	0.7105],
					[-0.7359,	0.6043],
					[-0.8807,	0.7105],
					[0.6017,	0.5019],
					[0.4747,	0.3996],
					[-0.6017,	0.5019],
					[-0.4747,	0.3996],
					[0.7359,	0.6043],
					[0.8807,	0.7105],
					[-0.7359,	0.6043],
					[-0.8807,	0.7105],
					[0.6017,	0.5019],
					[0.4747,	0.3996],
					[-0.6017,	0.5019],
					[-0.4747,	0.3996],
					[1.5441,	0.4545],
					[1.275,	0.3775],
					[-1.5441,	0.4545],
					[-1.275,	0.3775],
					[0.2586,	0.0752],
					[0.4435,	0.1065],
					[-0.2586,	0.0752],
					[-0.4435,	0.1065],
					[1.5441,	0.4545],
					[1.275,	0.3775],
					[-1.5441,	0.4545],
					[-1.275,	0.3775],
					[0.2586,	0.0752],
					[0.4435,	0.1065],
					[-0.2586,	0.0752],
					[-0.4435,	0.1065],
					[0.8925,	0.2771],
					[1.0649,	0.3219],
					[-0.8925,	0.2771],
					[-1.0649,	0.3219],
					[0.7362,	0.2279],
					[0.5936,	0.1699],
					[-0.7362,	0.2279],
					[-0.5936,	0.1699],
					[0.8925,	0.2771],
					[1.0649,	0.3219],
					[-0.8925,	0.2771],
					[-1.0649,	0.3219],
					[0.7362,	0.2279],
					[0.5936,	0.1699],
					[-0.7362,	0.2279],
					[-0.5936,	0.1699],
					[1.4352,	0.7452],
					[1.1866,	0.6182],
					[-1.4352,	0.7452],
					[-1.1866,	0.6182],
					[0.2523,	0.1944],
					[0.3695,	0.2695],
					[-0.2523,	0.1944],
					[-0.3695,	0.2695],
					[1.4352,	0.7452],
					[1.1866,	0.6182],
					[-1.4352,	0.7452],
					[-1.1866,	0.6182],
					[0.2523,	0.1944],
					[0.3695,	0.2695],
					[-0.2523,	0.1944],
					[-0.3695,	0.2695],
					[0.8273,	0.4493],
					[0.9911,	0.5243],
					[-0.8273,	0.4493],
					[-0.9911,	0.5243],
					[0.6708,	0.3859],
					[0.5197,	0.3331],
					[-0.6708,	0.3859],
					[-0.5197,	0.3331],
					[0.8273,	0.4493],
					[0.9911,	0.5243],
					[-0.8273,	0.4493],
					[-0.9911,	0.5243],
					[0.6708,	0.3859],
					[0.5197,	0.3331],
					[-0.6708,	0.3859],
					[-0.5197,	0.3331],
					[0.1646,	1.6329],
					[0.1379,	1.3595],
					[-0.1646,	1.6329],
					[-0.1379,	1.3595],
					[0.0736,	0.0898],
					[0.0742,	0.5054],
					[-0.0736,	0.0898],
					[-0.0742,	0.5054],
					[0.1646,	1.6329],
					[0.1379,	1.3595],
					[-0.1646,	1.6329],
					[-0.1379,	1.3595],
					[0.0736,	0.0898],
					[0.0742,	0.5054],
					[-0.0736,	0.0898],
					[-0.0742,	0.5054],
					[0.0992,	0.9847],
					[0.117,	1.1517],
					[-0.0992,	0.9847],
					[-0.117,	1.1517],
					[0.0894,	0.8287],
					[0.0889,	0.6739],
					[-0.0894,	0.8287],
					[-0.0889,	0.6739],
					[0.0992,	0.9847],
					[0.117,	1.1517],
					[-0.0992,	0.9847],
					[-0.117,	1.1517],
					[0.0894,	0.8287],
					[0.0889,	0.6739],
					[-0.0894,	0.8287],
					[-0.0889,	0.6739],
					[1.0516,	1.2481],
					[0.8742,	1.0355],
					[-1.0516,	1.2481],
					[-0.8742,	1.0355],
					[0.097,	0.245],
					[0.1959,	0.4045],
					[-0.097,	0.245],
					[-0.1959,	0.4045],
					[1.0516,	1.2481],
					[0.8742,	1.0355],
					[-1.0516,	1.2481],
					[-0.8742,	1.0355],
					[0.097,	0.245],
					[0.1959,	0.4045],
					[-0.097,	0.245],
					[-0.1959,	0.4045],
					[0.615,	0.7441],
					[0.7345,	0.8743],
					[-0.615,	0.7441],
					[-0.7345,	0.8743],
					[0.4932,	0.6301],
					[0.362,	0.5258],
					[-0.4932,	0.6301],
					[-0.362,	0.5258],
					[0.615,	0.7441],
					[0.7345,	0.8743],
					[-0.615,	0.7441],
					[-0.7345,	0.8743],
					[0.4932,	0.6301],
					[0.362,	0.5258],
					[-0.4932,	0.6301],
					[-0.362,	0.5258],
					[0.4866,	1.566],
					[0.4068,	1.3027],
					[-0.4866,	1.566],
					[-0.4068,	1.3027],
					[0.0732,	0.0899],
					[0.0877,	0.4997],
					[-0.0732,	0.0899],
					[-0.0877,	0.4997],
					[0.4866,	1.566],
					[0.4068,	1.3027],
					[-0.4866,	1.566],
					[-0.4068,	1.3027],
					[0.0732,	0.0899],
					[0.0877,	0.4997],
					[-0.0732,	0.0899],
					[-0.0877,	0.4997],
					[0.2927,	0.9409],
					[0.3446,	1.1023],
					[-0.2927,	0.9409],
					[-0.3446,	1.1023],
					[0.235,	0.7945],
					[0.167,	0.6529],
					[-0.235,	0.7945],
					[-0.167,	0.6529],
					[0.2927,	0.9409],
					[0.3446,	1.1023],
					[-0.2927,	0.9409],
					[-0.3446,	1.1023],
					[0.235,	0.7945],
					[0.167,	0.6529],
					[-0.235,	0.7945],
					[-0.167,	0.6529],
					[0.7867,	1.4356],
					[0.6561,	1.1927],
					[-0.7867,	1.4356],
					[-0.6561,	1.1927],
					[0.0947,	0.2451],
					[0.1865,	0.4121],
					[-0.0947,	0.2451],
					[-0.1865,	0.4121],
					[0.7867,	1.4356],
					[0.6561,	1.1927],
					[-0.7867,	1.4356],
					[-0.6561,	1.1927],
					[0.0947,	0.2451],
					[0.1865,	0.4121],
					[-0.0947,	0.2451],
					[-0.1865,	0.4121],
					[0.4677,	0.8579],
					[0.5537,	1.0081],
					[-0.4677,	0.8579],
					[-0.5537,	1.0081],
					[0.3893,	0.7143],
					[0.311,	0.5686],
					[-0.3893,	0.7143],
					[-0.311,	0.5686],
					[0.4677,	0.8579],
					[0.5537,	1.0081],
					[-0.4677,	0.8579],
					[-0.5537,	1.0081],
					[0.3893,	0.7143],
					[0.311,	0.5686],
					[-0.3893,	0.7143],
					[-0.311,	0.5686]]


# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["214"] = {"256APSK 3/4" : mod}




## 256APSK - 135/180
bits_per_symbol = 8


R1 = 1.0/5.2
R2 = 1.794/5.2
R3 = 2.409/5.2
R4 = 2.986/5.2
R5 = 3.579/5.2
R6 = 4.045/5.2
R7 = 4.5/5.2
R8 = 1.0

bit_map_phasor =   [[R1, 1*np.pi/32],
					[R1, 3*np.pi/32],
					[R1, 7*np.pi/32],
					[R1, 5*np.pi/32],
					[R1, 15*np.pi/32],
					[R1, 13*np.pi/32],
					[R1, 9*np.pi/32],
					[R1, 11*np.pi/32],
					[R1, 31*np.pi/32],
					[R1, 29*np.pi/32],
					[R1, 25*np.pi/32],
					[R1, 27*np.pi/32],
					[R1, 17*np.pi/32],
					[R1, 19*np.pi/32],
					[R1, 23*np.pi/32],
					[R1, 21*np.pi/32],
					[R1, 63*np.pi/32],
					[R1, 61*np.pi/32],
					[R1, 57*np.pi/32],
					[R1, 59*np.pi/32],
					[R1, 49*np.pi/32],
					[R1, 51*np.pi/32],
					[R1, 55*np.pi/32],
					[R1, 53*np.pi/32],
					[R1, 33*np.pi/32],
					[R1, 35*np.pi/32],
					[R1, 39*np.pi/32],
					[R1, 37*np.pi/32],
					[R1, 47*np.pi/32],
					[R1, 45*np.pi/32],
					[R1, 41*np.pi/32],
					[R1, 43*np.pi/32],
					[R2, 1*np.pi/32],
					[R2, 3*np.pi/32],
					[R2, 7*np.pi/32],
					[R2, 5*np.pi/32],
					[R2, 15*np.pi/32],
					[R2, 13*np.pi/32],
					[R2, 9*np.pi/32],
					[R2, 11*np.pi/32],
					[R2, 31*np.pi/32],
					[R2, 29*np.pi/32],
					[R2, 25*np.pi/32],
					[R2, 27*np.pi/32],
					[R2, 17*np.pi/32],
					[R2, 19*np.pi/32],
					[R2, 23*np.pi/32],
					[R2, 21*np.pi/32],
					[R2, 63*np.pi/32],
					[R2, 61*np.pi/32],
					[R2, 57*np.pi/32],
					[R2, 59*np.pi/32],
					[R2, 49*np.pi/32],
					[R2, 51*np.pi/32],
					[R2, 55*np.pi/32],
					[R2, 53*np.pi/32],
					[R2, 33*np.pi/32],
					[R2, 35*np.pi/32],
					[R2, 39*np.pi/32],
					[R2, 37*np.pi/32],
					[R2, 47*np.pi/32],
					[R2, 45*np.pi/32],
					[R2, 41*np.pi/32],
					[R2, 43*np.pi/32],
					[R4, 1*np.pi/32],
					[R4, 3*np.pi/32],
					[R4, 7*np.pi/32],
					[R4, 5*np.pi/32],
					[R4, 15*np.pi/32],
					[R4, 13*np.pi/32],
					[R4, 9*np.pi/32],
					[R4, 11*np.pi/32],
					[R4, 31*np.pi/32],
					[R4, 29*np.pi/32],
					[R4, 25*np.pi/32],
					[R4, 27*np.pi/32],
					[R4, 17*np.pi/32],
					[R4, 19*np.pi/32],
					[R4, 23*np.pi/32],
					[R4, 21*np.pi/32],
					[R4, 63*np.pi/32],
					[R4, 61*np.pi/32],
					[R4, 57*np.pi/32],
					[R4, 59*np.pi/32],
					[R4, 49*np.pi/32],
					[R4, 51*np.pi/32],
					[R4, 55*np.pi/32],
					[R4, 53*np.pi/32],
					[R4, 33*np.pi/32],
					[R4, 35*np.pi/32],
					[R4, 39*np.pi/32],
					[R4, 37*np.pi/32],
					[R4, 47*np.pi/32],
					[R4, 45*np.pi/32],
					[R4, 41*np.pi/32],
					[R4, 43*np.pi/32],
					[R3, 1*np.pi/32],
					[R3, 3*np.pi/32],
					[R3, 7*np.pi/32],
					[R3, 5*np.pi/32],
					[R3, 15*np.pi/32],
					[R3, 13*np.pi/32],
					[R3, 9*np.pi/32],
					[R3, 11*np.pi/32],
					[R3, 31*np.pi/32],
					[R3, 29*np.pi/32],
					[R3, 25*np.pi/32],
					[R3, 27*np.pi/32],
					[R3, 17*np.pi/32],
					[R3, 19*np.pi/32],
					[R3, 23*np.pi/32],
					[R3, 21*np.pi/32],
					[R3, 63*np.pi/32],
					[R3, 61*np.pi/32],
					[R3, 57*np.pi/32],
					[R3, 59*np.pi/32],
					[R3, 49*np.pi/32],
					[R3, 51*np.pi/32],
					[R3, 55*np.pi/32],
					[R3, 53*np.pi/32],
					[R3, 33*np.pi/32],
					[R3, 35*np.pi/32],
					[R3, 39*np.pi/32],
					[R3, 37*np.pi/32],
					[R3, 47*np.pi/32],
					[R3, 45*np.pi/32],
					[R3, 41*np.pi/32],
					[R3, 43*np.pi/32],
					[R8, 1*np.pi/32],
					[R8, 3*np.pi/32],
					[R8, 7*np.pi/32],
					[R8, 5*np.pi/32],
					[R8, 15*np.pi/32],
					[R8, 13*np.pi/32],
					[R8, 9*np.pi/32],
					[R8, 11*np.pi/32],
					[R8, 31*np.pi/32],
					[R8, 29*np.pi/32],
					[R8, 25*np.pi/32],
					[R8, 27*np.pi/32],
					[R8, 17*np.pi/32],
					[R8, 19*np.pi/32],
					[R8, 23*np.pi/32],
					[R8, 21*np.pi/32],
					[R8, 63*np.pi/32],
					[R8, 61*np.pi/32],
					[R8, 57*np.pi/32],
					[R8, 59*np.pi/32],
					[R8, 49*np.pi/32],
					[R8, 51*np.pi/32],
					[R8, 55*np.pi/32],
					[R8, 53*np.pi/32],
					[R8, 33*np.pi/32],
					[R8, 35*np.pi/32],
					[R8, 39*np.pi/32],
					[R8, 37*np.pi/32],
					[R8, 47*np.pi/32],
					[R8, 45*np.pi/32],
					[R8, 41*np.pi/32],
					[R8, 43*np.pi/32],
					[R7, 1*np.pi/32],
					[R7, 3*np.pi/32],
					[R7, 7*np.pi/32],
					[R7, 5*np.pi/32],
					[R7, 15*np.pi/32],
					[R7, 13*np.pi/32],
					[R7, 9*np.pi/32],
					[R7, 11*np.pi/32],
					[R7, 31*np.pi/32],
					[R7, 29*np.pi/32],
					[R7, 25*np.pi/32],
					[R7, 27*np.pi/32],
					[R7, 17*np.pi/32],
					[R7, 19*np.pi/32],
					[R7, 23*np.pi/32],
					[R7, 21*np.pi/32],
					[R7, 63*np.pi/32],
					[R7, 61*np.pi/32],
					[R7, 57*np.pi/32],
					[R7, 59*np.pi/32],
					[R7, 49*np.pi/32],
					[R7, 51*np.pi/32],
					[R7, 55*np.pi/32],
					[R7, 53*np.pi/32],
					[R7, 33*np.pi/32],
					[R7, 35*np.pi/32],
					[R7, 39*np.pi/32],
					[R7, 37*np.pi/32],
					[R7, 47*np.pi/32],
					[R7, 45*np.pi/32],
					[R7, 41*np.pi/32],
					[R7, 43*np.pi/32],
					[R5, 1*np.pi/32],
					[R5, 3*np.pi/32],
					[R5, 7*np.pi/32],
					[R5, 5*np.pi/32],
					[R5, 15*np.pi/32],
					[R5, 13*np.pi/32],
					[R5, 9*np.pi/32],
					[R5, 11*np.pi/32],
					[R5, 31*np.pi/32],
					[R5, 29*np.pi/32],
					[R5, 25*np.pi/32],
					[R5, 27*np.pi/32],
					[R5, 17*np.pi/32],
					[R5, 19*np.pi/32],
					[R5, 23*np.pi/32],
					[R5, 21*np.pi/32],
					[R5, 63*np.pi/32],
					[R5, 61*np.pi/32],
					[R5, 57*np.pi/32],
					[R5, 59*np.pi/32],
					[R5, 49*np.pi/32],
					[R5, 51*np.pi/32],
					[R5, 55*np.pi/32],
					[R5, 53*np.pi/32],
					[R5, 33*np.pi/32],
					[R5, 35*np.pi/32],
					[R5, 39*np.pi/32],
					[R5, 37*np.pi/32],
					[R5, 47*np.pi/32],
					[R5, 45*np.pi/32],
					[R5, 41*np.pi/32],
					[R5, 43*np.pi/32],
					[R6, 1*np.pi/32],
					[R6, 3*np.pi/32],
					[R6, 7*np.pi/32],
					[R6, 5*np.pi/32],
					[R6, 15*np.pi/32],
					[R6, 13*np.pi/32],
					[R6, 9*np.pi/32],
					[R6, 11*np.pi/32],
					[R6, 31*np.pi/32],
					[R6, 29*np.pi/32],
					[R6, 25*np.pi/32],
					[R6, 27*np.pi/32],
					[R6, 17*np.pi/32],
					[R6, 19*np.pi/32],
					[R6, 23*np.pi/32],
					[R6, 21*np.pi/32],
					[R6, 63*np.pi/32],
					[R6, 61*np.pi/32],
					[R6, 57*np.pi/32],
					[R6, 59*np.pi/32],
					[R6, 49*np.pi/32],
					[R6, 51*np.pi/32],
					[R6, 55*np.pi/32],
					[R6, 53*np.pi/32],
					[R6, 33*np.pi/32],
					[R6, 35*np.pi/32],
					[R6, 39*np.pi/32],
					[R6, 37*np.pi/32],
					[R6, 47*np.pi/32],
					[R6, 45*np.pi/32],
					[R6, 41*np.pi/32],
					[R6, 43*np.pi/32]]

# convert to cartesion 
bit_map_cartesian = [[np.real(_[0]*np.exp(1j*_[1])), np.imag(_[0]*np.exp(1j*_[1]))] for _ in bit_map_phasor] 



# create dictionary
mod = {	"bits_per_symbol"	:	bits_per_symbol,
		"offset"			:	False,
		"filter"			:	"RRC",
		"relative_rate"		:	1.0,
		"bit_map"			:	bit_map_cartesian}

# add dictionary to main constellations dictionary
constellations["214"] = {"256APSK 3/4" : mod}


# write the data to a JSON file
with open(json_filename, 'w') as outfile:
    json.dump(constellations, outfile)