
VERILOG_SOURCES = 	$(PWD)/../apsk_modulator.v \
					$(PWD)/../apsk_modulator_control.v \
					$(PWD)/../../lookup_table/lookup_table.v \
					$(PWD)/../../lookup_table_behavioural/lookup_table_behavioural.v \
					$(PWD)/../../polyphase_filter/polyphase_filter.v \
					$(PWD)/../../multiply_accumulate/multiply_accumulate.v \
					$(PWD)/../../multiply_accumulate_behavioural/multiply_accumulate_behavioural.v \
					$(PWD)/../../fir_filter/fir_filter.v \
					$(XILINX_VERILOG_LIB)/src/glbl.v
					
COMPILE_ARGS = -s glbl -y $(XILINX_VERILOG_LIB)/src/unisims

TOPLEVEL = apsk_modulator
MODULE = test_apsk_modulator

include $(shell cocotb-config --makefiles)/Makefile.inc
include $(shell cocotb-config --makefiles)/Makefile.sim

wave:
	gtkwave waveform.vcd waveform.gtkw