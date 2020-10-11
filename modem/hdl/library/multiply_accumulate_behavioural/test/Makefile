

VERILOG_SOURCES = 	$(PWD)/../multiply_accumulate_behavioural.v

TOPLEVEL = multiply_accumulate_behavioural
MODULE = test_multiply_accumulate_behavioural

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw
