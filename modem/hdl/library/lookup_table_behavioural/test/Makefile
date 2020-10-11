

VERILOG_SOURCES = 	$(PWD)/../lookup_table_behavioural.v

TOPLEVEL = lookup_table_behavioural
MODULE = test_lookup_table_behavioural

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw
