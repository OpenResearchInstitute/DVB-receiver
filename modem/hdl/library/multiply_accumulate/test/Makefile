

VERILOG_SOURCES = 	$(PWD)/../multiply_accumulate.v \
					$(XILINX_VERILOG_LIB)/src/glbl.v
					
COMPILE_ARGS = -s glbl -y $(XILINX_VERILOG_LIB)/src/unisims

TOPLEVEL = multiply_accumulate
MODULE = test_multiply_accumulate

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw