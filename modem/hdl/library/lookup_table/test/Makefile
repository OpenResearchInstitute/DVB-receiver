

VERILOG_SOURCES = 	$(PWD)/../lookup_table.v \
					$(XILINX_VERILOG_LIB)/src/glbl.v
					
COMPILE_ARGS = -s glbl -y $(XILINX_VERILOG_LIB)/src/unisims

TOPLEVEL = lookup_table
MODULE = test_lookup_table

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw