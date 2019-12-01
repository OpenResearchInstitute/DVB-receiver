

VERILOG_SOURCES = 	$(PWD)/../fir_filter.v \
					$(PWD)/../../multiply_accumulate/multiply_accumulate.v \
					$(XILINX_VERILOG_LIB)/src/glbl.v
					
COMPILE_ARGS = -s glbl -y $(XILINX_VERILOG_LIB)/src/unisims

TOPLEVEL = fir_filter
MODULE = test_fir_filter

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw