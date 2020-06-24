
//
/* 
 * Copyright 2020 <+YOU OR YOUR COMPANY+>.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

//
module noc_block_apskmodulator #(
  parameter NOC_ID = 64'h1F860BEBEC67AA56,
  parameter STR_SINK_FIFOSIZE = 11,
  parameter BITS_PER_SYMBOL_WIDTH = 4)
(
  input bus_clk, input bus_rst,
  input ce_clk, input ce_rst,
  input  [63:0] i_tdata, input  i_tlast, input  i_tvalid, output i_tready,
  output [63:0] o_tdata, output o_tlast, output o_tvalid, input  o_tready,
  output [63:0] debug
);

  ////////////////////////////////////////////////////////////
  //
  // RFNoC Shell
  //
  ////////////////////////////////////////////////////////////
  wire [31:0] set_data;
  wire [7:0]  set_addr;
  wire        set_stb;
  reg  [63:0] rb_data;
  wire [7:0]  rb_addr;

  wire [63:0] cmdout_tdata, ackin_tdata;
  wire        cmdout_tlast, cmdout_tvalid, cmdout_tready, ackin_tlast, ackin_tvalid, ackin_tready;

  wire [63:0] str_sink_tdata, str_src_tdata;
  wire        str_sink_tlast, str_sink_tvalid, str_sink_tready, str_src_tlast, str_src_tvalid, str_src_tready;

  wire [15:0] src_sid;
  wire [15:0] next_dst_sid, resp_out_dst_sid;
  wire [15:0] resp_in_dst_sid;

  wire        clear_tx_seqnum;

  noc_shell #(
    .NOC_ID(NOC_ID),
    .STR_SINK_FIFOSIZE(STR_SINK_FIFOSIZE))
  noc_shell (
    .bus_clk(bus_clk), .bus_rst(bus_rst),
    .i_tdata(i_tdata), .i_tlast(i_tlast), .i_tvalid(i_tvalid), .i_tready(i_tready),
    .o_tdata(o_tdata), .o_tlast(o_tlast), .o_tvalid(o_tvalid), .o_tready(o_tready),
    // Computer Engine Clock Domain
    .clk(ce_clk), .reset(ce_rst),
    // Control Sink
    .set_data(set_data), .set_addr(set_addr), .set_stb(set_stb), .set_time(), .set_has_time(),
    .rb_stb(1'b1), .rb_data(rb_data), .rb_addr(rb_addr),
    // Control Source
    .cmdout_tdata(cmdout_tdata), .cmdout_tlast(cmdout_tlast), .cmdout_tvalid(cmdout_tvalid), .cmdout_tready(cmdout_tready),
    .ackin_tdata(ackin_tdata), .ackin_tlast(ackin_tlast), .ackin_tvalid(ackin_tvalid), .ackin_tready(ackin_tready),
    // Stream Sink
    .str_sink_tdata(str_sink_tdata), .str_sink_tlast(str_sink_tlast), .str_sink_tvalid(str_sink_tvalid), .str_sink_tready(str_sink_tready),
    // Stream Source
    .str_src_tdata(str_src_tdata), .str_src_tlast(str_src_tlast), .str_src_tvalid(str_src_tvalid), .str_src_tready(str_src_tready),
    // Stream IDs set by host
    .src_sid(src_sid),                   // SID of this block
    .next_dst_sid(next_dst_sid),         // Next destination SID
    .resp_in_dst_sid(resp_in_dst_sid),   // Response destination SID for input stream responses / errors
    .resp_out_dst_sid(resp_out_dst_sid), // Response destination SID for output stream responses / errors
    // Misc
    .vita_time('d0), .clear_tx_seqnum(clear_tx_seqnum),
    .debug(debug));

  ////////////////////////////////////////////////////////////
  //
  // AXI Wrapper
  // Convert RFNoC Shell interface into AXI stream interface
  //
  ////////////////////////////////////////////////////////////
  localparam AXI_WIDTH    =   32;


  wire [31:0] m_axis_data_tdata;
  wire        m_axis_data_tlast;
  wire        m_axis_data_tvalid;
  wire        m_axis_data_tready;

  wire [31:0] s_axis_data_tdata;
  wire        s_axis_data_tlast;
  wire        s_axis_data_tvalid;
  wire        s_axis_data_tready;

  axi_wrapper #(
    .SIMPLE_MODE(1))
  axi_wrapper (
    .clk(ce_clk), .reset(ce_rst),
    .bus_clk(bus_clk), .bus_rst(bus_rst),
    .clear_tx_seqnum(clear_tx_seqnum),
    .next_dst(next_dst_sid),
    .set_stb(set_stb), .set_addr(set_addr), .set_data(set_data),
    .i_tdata(str_sink_tdata), .i_tlast(str_sink_tlast), .i_tvalid(str_sink_tvalid), .i_tready(str_sink_tready),
    .o_tdata(str_src_tdata), .o_tlast(str_src_tlast), .o_tvalid(str_src_tvalid), .o_tready(str_src_tready),
    .m_axis_data_tdata(m_axis_data_tdata),
    .m_axis_data_tlast(m_axis_data_tlast),
    .m_axis_data_tvalid(m_axis_data_tvalid),
    .m_axis_data_tready(m_axis_data_tready),
    .m_axis_data_tuser(),
    .s_axis_data_tdata(s_axis_data_tdata),
    .s_axis_data_tlast(s_axis_data_tlast),
    .s_axis_data_tvalid(s_axis_data_tvalid),
    .s_axis_data_tready(s_axis_data_tready),
    .s_axis_data_tuser(),
    .m_axis_config_tdata(),
    .m_axis_config_tlast(),
    .m_axis_config_tvalid(),
    .m_axis_config_tready(),
    .m_axis_pkt_len_tdata(),
    .m_axis_pkt_len_tvalid(),
    .m_axis_pkt_len_tready());

  ////////////////////////////////////////////////////////////
  //
  // User code
  //
  ////////////////////////////////////////////////////////////
  // NoC Shell registers 0 - 127,
  // User register address space starts at 128
  localparam SR_USER_REG_BASE = 128;

  // Control Source Unused
  assign cmdout_tdata  = 64'd0;
  assign cmdout_tlast  = 1'b0;
  assign cmdout_tvalid = 1'b0;
  assign ackin_tready  = 1'b1;

  // Settings registers
  //
  // - The settings register bus is a simple strobed interface.
  // - Transactions include both a write and a readback.
  // - The write occurs when set_stb is asserted.
  //   The settings register with the address matching set_addr will
  //   be loaded with the data on set_data.
  // - Readback occurs when rb_stb is asserted. The read back strobe
  //   must assert at least one clock cycle after set_stb asserts /
  //   rb_stb is ignored if asserted on the same clock cycle of set_stb.
  //   Example valid and invalid timing:
  //              __    __    __    __
  //   clk     __|  |__|  |__|  |__|  |__
  //               _____
  //   set_stb ___|     |________________
  //                     _____
  //   rb_stb  _________|     |__________     (Valid)
  //                           _____
  //   rb_stb  _______________|     |____     (Valid)
  //           __________________________
  //   rb_stb                                 (Valid if readback data is a constant)
  //               _____
  //   rb_stb  ___|     |________________     (Invalid / ignored, same cycle as set_stb)
  //
  localparam [7:0] SR_BITS_PER_SYMBOL = SR_USER_REG_BASE;
  localparam [7:0] SR_OFFSET_SYMBOL_ENABLE = SR_USER_REG_BASE + 8'd1;

  wire [BITS_PER_SYMBOL_WIDTH-1:0] bits_per_symbol_reg;
  setting_reg #(
    .my_addr(SR_BITS_PER_SYMBOL), .awidth(8), .width(BITS_PER_SYMBOL_WIDTH))
  bits_per_symbol_reg_inst (
    .clk(ce_clk), .rst(ce_rst),
    .strobe(set_stb), .addr(set_addr), .in(set_data), .out(bits_per_symbol_reg), .changed());

  wire [0:0] offset_symbol_enable_reg;
  setting_reg #(
    .my_addr(SR_OFFSET_SYMBOL_ENABLE), .awidth(8), .width(1))
  offset_symbol_enable_reg_inst (
    .clk(ce_clk), .rst(ce_rst),
    .strobe(set_stb), .addr(set_addr), .in(set_data), .out(offset_symbol_enable_reg), .changed());

  // Readback registers
  // rb_stb set to 1'b1 on NoC Shell
  always @(posedge ce_clk) begin
    case(rb_addr)
      8'd0 : rb_data <= {32'd0, bits_per_symbol_reg};
      8'd1 : rb_data <= {32'd0, offset_symbol_enable_reg};
      default : rb_data <= 64'h0BADC0DE0BADC0DE;
    endcase
  end


  //////////////////////////////////////////////////
  //
  // AXI Configuration interfaces
  //
  //////////////////////////////////////////////////
  wire                      coefficients_in_tready;
  wire  [AXI_WIDTH-1:0]     coefficients_in_tdata;
  wire                      coefficients_in_tlast;
  wire                      coefficients_in_tvalid;

  wire                      lut_data_load_tready;
  wire  [AXI_WIDTH-1:0]     lut_data_load_tdata;
  wire                      lut_data_load_tlast;
  wire                      lut_data_load_tvalid;



  localparam SR_COEFFS       = 130;
  localparam SR_COEFFS_TLAST = 131;
  localparam COEFF_WIDTH     = 32;

  localparam SR_CONSTS       = 132;
  localparam SR_CONSTS_TLAST = 133;
  localparam CONSTS_WIDTH    = 32;


  // Pulse filter coefficient reload bus
  axi_setting_reg #(
    .ADDR(SR_COEFFS),
    .USE_ADDR_LAST(1),
    .ADDR_LAST(SR_COEFFS_TLAST),
    .WIDTH(COEFF_WIDTH))
  set_pulse_coeff (
    .clk(ce_clk),
    .reset(ce_rst),
    .set_stb(set_stb),
    .set_addr(set_addr),
    .set_data(set_data),
    .o_tdata(coefficients_in_tdata),
    .o_tlast(coefficients_in_tlast),
    .o_tvalid(coefficients_in_tvalid),
    .o_tready(coefficients_in_tready));

  // Constellation map reload bus
  axi_setting_reg #(
    .ADDR(SR_CONSTS),
    .USE_ADDR_LAST(1),
    .ADDR_LAST(SR_CONSTS_TLAST),
    .WIDTH(CONSTS_WIDTH))
  set_const_map (
    .clk(ce_clk),
    .reset(ce_rst),
    .set_stb(set_stb),
    .set_addr(set_addr),
    .set_data(set_data),
    .o_tdata(lut_data_load_tdata),
    .o_tlast(lut_data_load_tlast),
    .o_tvalid(lut_data_load_tvalid),
    .o_tready(lut_data_load_tready));


  apsk_modulator #(
    .SAMPLES_PER_SYMBOL(4),
    .NUMBER_TAPS(40),
    .DATA_WIDTH(16),
    .COEFFICIENT_WIDTH(16),
    .BITS_PER_SYMBOL_WIDTH(BITS_PER_SYMBOL_WIDTH),
    .DATA_IN_TDATA_WIDTH(32),
    .DATA_OUT_TDATA_WIDTH(32),
    .LUT_DATA_LOAD_TDATA_WIDTH(32),
    .COEFFICIENTS_IN_TDATA_WIDTH(32),
    .CONTROL_DATA_WIDTH(32),
    .CONTROL_ADDR_WIDTH(4)
  ) apsk_modulator_inst (
    // data input AXI bus
    .data_in_aclk(ce_clk),
    .data_in_aresetn(~ce_rst),
    .data_in_tready(m_axis_data_tready),
    .data_in_tdata(m_axis_data_tdata),
    .data_in_tlast(m_axis_data_tlast),
    .data_in_tvalid(m_axis_data_tvalid),

    // data output AXI bus
    .data_out_aclk(ce_clk),
    .data_out_aresetn(~ce_rst),
    .data_out_tready(s_axis_data_tready),
    .data_out_tdata(s_axis_data_tdata),
    .data_out_tlast(s_axis_data_tlast),
    .data_out_tvalid(s_axis_data_tvalid),

    // lookup table load AXI bus
    .lut_data_load_aclk(ce_clk),
    .lut_data_load_aresetn(~ce_rst),
    .lut_data_load_tready(lut_data_load_tready),
    .lut_data_load_tdata(lut_data_load_tdata),
    .lut_data_load_tlast(lut_data_load_tlast),
    .lut_data_load_tvalid(lut_data_load_tvalid),

    // pulse shaping filter load AXI bus
    .coefficients_in_aclk(ce_clk),
    .coefficients_in_aresetn(~ce_rst),
    .coefficients_in_tready(coefficients_in_tready),
    .coefficients_in_tdata(coefficients_in_tdata),
    .coefficients_in_tlast(coefficients_in_tlast),
    .coefficients_in_tvalid(coefficients_in_tvalid),

    // control signals
    .bits_per_symbol(bits_per_symbol_reg),
    .offset_symbol_enable(offset_symbol_enable_reg)
  );

endmodule
