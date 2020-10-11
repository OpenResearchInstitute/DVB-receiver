
`timescale 1 ns / 1 ps

	module apsk_modulator #
	(
		// block setup information
		parameter integer SAMPLES_PER_SYMBOL			= 4,
		parameter integer NUMBER_TAPS					= 40,
		parameter integer DATA_WIDTH					= 16,
		parameter integer COEFFICIENT_WIDTH				= 16,

		// AXI buses parameters
		parameter integer DATA_IN_TDATA_WIDTH			= 32,
		parameter integer DATA_OUT_TDATA_WIDTH			= 32,
		parameter integer LUT_DATA_LOAD_TDATA_WIDTH		= 32,
		parameter integer COEFFICIENTS_IN_TDATA_WIDTH	= 32,
		parameter integer CONTROL_DATA_WIDTH			= 32,
		parameter integer CONTROL_ADDR_WIDTH			= 4
	)
	(
		// data input AXI bus
		input wire  										data_in_aclk,
		input wire  										data_in_aresetn,
		output wire											data_in_tready,
		input wire 	[DATA_IN_TDATA_WIDTH-1:0] 				data_in_tdata,
		input wire  										data_in_tlast,
		input wire  										data_in_tvalid,

		// data output AXI bus
		input wire  										data_out_aclk,
		input wire  										data_out_aresetn,
		input wire 											data_out_tready,
		output wire [DATA_OUT_TDATA_WIDTH-1:0] 				data_out_tdata,
		output wire 										data_out_tlast,
		output wire 										data_out_tvalid,

		// lookup table load AXI bus
		input wire  										lut_data_load_aclk,
		input wire  										lut_data_load_aresetn,
		output wire 										lut_data_load_tready,
		input wire 	[LUT_DATA_LOAD_TDATA_WIDTH-1:0] 		lut_data_load_tdata,
		input wire  										lut_data_load_tlast,
		input wire  										lut_data_load_tvalid,

		// pulse shaping filter load AXI bus
		input wire  										coefficients_in_aclk,
		input wire  										coefficients_in_aresetn,
		output wire 										coefficients_in_tready,
		input wire 	[COEFFICIENTS_IN_TDATA_WIDTH-1:0] 		coefficients_in_tdata,
		input wire  										coefficients_in_tlast,
		input wire  										coefficients_in_tvalid,

		// control signal AXI bus
		input wire  										control_aclk,
		input wire  										control_aresetn,
		input wire 	[CONTROL_ADDR_WIDTH-1:0] 				control_awaddr,
		input wire 	[2:0] 									control_awprot,
		input wire  										control_awvalid,
		output wire 										control_awready,
		input wire 	[CONTROL_DATA_WIDTH-1:0] 				control_wdata,
		input wire 	[(CONTROL_DATA_WIDTH/8)-1:0] 			control_wstrb,
		input wire  										control_wvalid,
		output wire 										control_wready,
		output wire [1:0] 									control_bresp,
		output wire 										control_bvalid,
		input wire  										control_bready,
		input wire 	[CONTROL_ADDR_WIDTH-1:0] 				control_araddr,
		input wire 	[2:0] 									control_arprot,
		input wire  										control_arvalid,
		output wire  										control_arready,
		output wire [CONTROL_DATA_WIDTH-1:0] 				control_rdata,
		output wire [1:0] 									control_rresp,
		output wire 										control_rvalid,
		input wire  										control_rready
	);

	// define some constants
	localparam integer ADDRESS_WIDTH 			= 8;
	localparam integer DATA_LATCH_EXCESS 		= 8;
	localparam integer DATA_LATCH_WIDTH 		= DATA_IN_TDATA_WIDTH + DATA_LATCH_EXCESS;
	localparam integer BITS_PER_SYMBOL_WIDTH 	= $clog2(ADDRESS_WIDTH+1);

	wire  									reset;

	// data in signals
	reg [$clog2(DATA_IN_TDATA_WIDTH):0]		data_in_count;
	reg [DATA_LATCH_WIDTH-1:0]				data_in_latch;
	reg [7:0]								data_in_latch_mask;
	reg 									data_in_tlast_latched;


	// lookup table signals
	wire  									lut_data_in_aclk;
	wire  									lut_data_in_aresetn;
	wire									lut_data_in_tready;
	wire [ADDRESS_WIDTH-1:0] 				lut_data_in_tdata;
	reg  									lut_data_in_tlast;
	reg  									lut_data_in_tvalid;
	wire  									lut_data_out_aclk;
	wire  									lut_data_out_aresetn;
	wire									lut_data_out_tready_i;
	wire									lut_data_out_tready_q;
	wire [2*DATA_WIDTH-1:0]					lut_data_out_tdata;
	wire  									lut_data_out_tlast;
	wire  									lut_data_out_tvalid;

	// filter signals
	wire 									filter_i_out_aclk;
	wire 									filter_q_out_aclk;
	wire 									filter_i_out_aresetn;
	wire 									filter_q_out_aresetn;
	wire 									filter_i_out_tready;
	wire 									filter_q_out_tready;
	wire [DATA_WIDTH-1:0]					filter_i_out_tdata;
	wire [DATA_WIDTH-1:0]					filter_q_out_tdata;
	wire 									filter_i_out_tlast;
	wire 									filter_q_out_tlast;
	wire 									filter_i_out_tvalid;
	wire 									filter_q_out_tvalid;
	wire 									coefficients_i_in_tready;
	wire 									coefficients_q_in_tready;

	// configuration signals
	wire [3:0] 								bits_per_symbol;
	wire 									offset_symbol_enable;


	// form an internal reset signal which can be controled through external ports
	//   or reset at the end of a transaction
	assign reset = !data_in_aresetn | data_out_tlast;


	// map the clocks and resets
	assign lut_data_in_aclk = data_in_aclk;
	assign lut_data_out_aclk = data_in_aclk;
	assign filter_i_out_aclk = data_in_aclk;
	assign filter_q_out_aclk = data_in_aclk;
	
	assign lut_data_in_aresetn = !reset;
	assign lut_data_out_aresetn = !reset;
	//assign lut_data_load_aresetn = data_in_aresetn;
	assign filter_i_out_aresetn = !reset;
	assign filter_q_out_aresetn = !reset;


	// figure out if the input is currently valid
	always @(posedge data_in_aclk) begin
		if(reset) begin
			lut_data_in_tvalid <= 0;
		end
		else begin

			// valid input count and we're in the correct phase so the data into the
			//  lookup table is valid
			if ((data_in_count > 0) | data_in_tvalid) begin
				lut_data_in_tvalid <= 1;
			end

			// we're finished with the frame so removed valid flag
			else if (lut_data_in_tlast & lut_data_in_tready & (data_in_count == 0)) begin
				lut_data_in_tvalid <= 0;
			end

			// register the signal
			else begin
				lut_data_in_tvalid <= lut_data_in_tvalid;
			end
		end
	end

	
	// if the data input count has reached 
	assign data_in_tready = !(data_in_count > bits_per_symbol) ? lut_data_in_tready & !data_in_tlast_latched : 0;


	// count how many bits are left in the input shift register
	always @(posedge data_in_aclk) begin
		if(reset) begin
			data_in_count <= 0;
		end
		else begin

			// new data, reset count
			if ((data_in_count <= bits_per_symbol) & data_in_tready & data_in_tvalid) begin
				data_in_count <= DATA_IN_TDATA_WIDTH - bits_per_symbol + data_in_count;
			end

			// need to keep clearing the current data
			else if (lut_data_out_tready & lut_data_out_tvalid) begin
				data_in_count <= data_in_count - bits_per_symbol;
			end

			// register the count
			else begin
				data_in_count <= data_in_count;
			end
		end
	end

	// handle the last sample signal
	always @(posedge data_in_aclk) begin
		if(reset) begin
			lut_data_in_tlast <= 0;
			data_in_tlast_latched <= 0;
		end
		else begin

			// latch the tlast signal
			data_in_tlast_latched = data_in_tlast_latched | (data_in_tlast & data_in_tvalid & data_in_tready);

			// if at the end of the shifting out data then signal last sample
			//  and clear latch
			if (!(data_in_count >= bits_per_symbol) & !data_in_tlast) begin
				lut_data_in_tlast <= data_in_tlast_latched;
			end

			// reset the tlast latch once the transaction is complete
			else if (data_out_tlast) begin
				data_in_tlast_latched <= 0;
			end

			// register the signal
			else begin
				lut_data_in_tlast <= lut_data_in_tlast;
			end
		end
	end

	// latch in data and shift through it splitting into bit segments
	always @(posedge data_in_aclk) begin
		if(reset) begin
			data_in_latch <= 0;
		end
		else begin

			// new input so latch it into the shift register
			if ((data_in_count <= bits_per_symbol) & data_in_tready & data_in_tvalid) begin

				// blocking shift of the register before mapping - may not be a very good may to perform this
				data_in_latch = data_in_latch >> bits_per_symbol;

				// figure out how to keep the requried samples left over
				case (data_in_count)
					0  : data_in_latch <= {8'd0, data_in_tdata};
					1  : data_in_latch <= {7'd0, data_in_tdata, data_in_latch[0]};
					2  : data_in_latch <= {6'd0, data_in_tdata, data_in_latch[1:0]};
					3  : data_in_latch <= {5'd0, data_in_tdata, data_in_latch[2:0]};
					4  : data_in_latch <= {4'd0, data_in_tdata, data_in_latch[3:0]};
					5  : data_in_latch <= {3'd0, data_in_tdata, data_in_latch[4:0]};
					6  : data_in_latch <= {2'd0, data_in_tdata, data_in_latch[5:0]};
					7  : data_in_latch <= {1'd0, data_in_tdata, data_in_latch[6:0]};
					8  : data_in_latch <= {data_in_tdata, data_in_latch[7:0]};
					default : data_in_latch <= {8'd0, data_in_tdata};
				endcase
			end

			// shift the input register to obtain fresh information at the bottom
			else if (lut_data_out_tready & lut_data_out_tvalid) begin
				data_in_latch <= data_in_latch >> bits_per_symbol;
			end

			// register the signal
			else begin
				data_in_latch <= data_in_latch;
			end
		end
	end


	// select a bit mask depending on the number of bits of symbol
	// TODO should be change from hardcoded
	always @(*)
	begin
		case (bits_per_symbol)
			0  : data_in_latch_mask <= 8'b00000001;
			1  : data_in_latch_mask <= 8'b00000001;
			2  : data_in_latch_mask <= 8'b00000011;
			3  : data_in_latch_mask <= 8'b00000111;
			4  : data_in_latch_mask <= 8'b00001111;
			5  : data_in_latch_mask <= 8'b00011111;
			6  : data_in_latch_mask <= 8'b00111111;
			7  : data_in_latch_mask <= 8'b01111111;
			8  : data_in_latch_mask <= 8'b11111111;
			default : data_in_latch_mask <= 8'b00000001; 
		endcase
	end


	// interface to the lookup table
	assign lut_data_in_tdata = data_in_latch[ADDRESS_WIDTH-1:0] & data_in_latch_mask;
	


	// instantiate the modulation lookup table
	lookup_table_behavioural #(
		.TDATA_WIDTH(2*DATA_WIDTH),
		.ADDRESS_WIDTH(ADDRESS_WIDTH)
	) lookup_table_inst (
		.data_in_aclk(lut_data_in_aclk),
		.data_in_aresetn(lut_data_in_aresetn),
		.data_in_tready(lut_data_in_tready),
		.data_in_tdata(lut_data_in_tdata),
		.data_in_tlast(lut_data_in_tlast),
		.data_in_tvalid(lut_data_in_tvalid),
		.data_out_aclk(lut_data_out_aclk),
		.data_out_aresetn(lut_data_out_aresetn),
		.data_out_tready(lut_data_out_tready),
		.data_out_tdata(lut_data_out_tdata),
		.data_out_tlast(lut_data_out_tlast),
		.data_out_tvalid(lut_data_out_tvalid),
		.data_load_aclk(lut_data_load_aclk),
		.data_load_aresetn(lut_data_load_aresetn),
		.data_load_tready(lut_data_load_tready),
		.data_load_tdata(lut_data_load_tdata),
		.data_load_tlast(lut_data_load_tlast),
		.data_load_tvalid(lut_data_load_tvalid)
	);

	// both the pulse shaping filters need to be ready
	assign lut_data_out_tready = lut_data_out_tready_i & lut_data_out_tready_q;


	// combine the neccesary i and q signals
	assign coefficients_in_tready = coefficients_i_in_tready & coefficients_i_in_tready;
	assign filter_i_out_tready = data_out_tready;
	assign filter_q_out_tready = data_out_tready;


	// I channel polyphase pulse shaping filter
	polyphase_filter #(
		.NUMBER_TAPS(NUMBER_TAPS),
		.DATA_IN_WIDTH(DATA_WIDTH),
		.DATA_OUT_WIDTH(DATA_WIDTH),
		.COEFFICIENT_WIDTH(COEFFICIENT_WIDTH),
		.RATE_CHANGE(SAMPLES_PER_SYMBOL),
		.DECIMATE_INTERPOLATE(1)
	) pulse_shape_filter_i (
		.data_in_aclk(lut_data_out_aclk),
		.data_in_aresetn(lut_data_out_aresetn),
		.data_in_tready(lut_data_out_tready_i),
		.data_in_tdata(lut_data_out_tdata[2*DATA_WIDTH-1:DATA_WIDTH]),
		.data_in_tlast(lut_data_out_tlast),
		.data_in_tvalid(lut_data_out_tvalid),
		.data_out_aclk(filter_i_out_aclk),
		.data_out_aresetn(filter_i_out_aresetn),
		.data_out_tready(filter_i_out_tready),
		.data_out_tdata(filter_i_out_tdata),
		.data_out_tlast(filter_i_out_tlast),
		.data_out_tvalid(filter_i_out_tvalid),
		.coefficients_in_aclk(coefficients_in_aclk),
		.coefficients_in_aresetn(coefficients_in_aresetn),
		.coefficients_in_tready(coefficients_i_in_tready),
		.coefficients_in_tdata(coefficients_in_tdata[COEFFICIENT_WIDTH-1:0]),
		.coefficients_in_tlast(coefficients_in_tlast),
		.coefficients_in_tvalid(coefficients_in_tvalid)
	);

	// Q channel polyphase pulse shaping filter
	polyphase_filter #(
		.NUMBER_TAPS(NUMBER_TAPS),
		.DATA_IN_WIDTH(DATA_WIDTH),
		.DATA_OUT_WIDTH(DATA_WIDTH),
		.COEFFICIENT_WIDTH(COEFFICIENT_WIDTH),
		.RATE_CHANGE(SAMPLES_PER_SYMBOL),
		.DECIMATE_INTERPOLATE(1)
	) pulse_shape_filter_q (
		.data_in_aclk(lut_data_out_aclk),
		.data_in_aresetn(lut_data_out_aresetn),
		.data_in_tready(lut_data_out_tready_q),
		.data_in_tdata(lut_data_out_tdata[DATA_WIDTH-1:0]),
		.data_in_tlast(lut_data_out_tlast),
		.data_in_tvalid(lut_data_out_tvalid),
		.data_out_aclk(filter_q_out_aclk),
		.data_out_aresetn(filter_q_out_aresetn),
		.data_out_tready(filter_q_out_tready),
		.data_out_tdata(filter_q_out_tdata),
		.data_out_tlast(filter_q_out_tlast),
		.data_out_tvalid(filter_q_out_tvalid),
		.coefficients_in_aclk(coefficients_in_aclk),
		.coefficients_in_aresetn(coefficients_in_aresetn),
		.coefficients_in_tready(coefficients_q_in_tready),
		.coefficients_in_tdata(coefficients_in_tdata[COEFFICIENT_WIDTH-1:0]),
		.coefficients_in_tlast(coefficients_in_tlast),
		.coefficients_in_tvalid(coefficients_in_tvalid)
	);

	// connect the pulse shaping filter to the output
	assign data_out_tdata = {filter_i_out_tdata, filter_q_out_tdata};
	assign data_out_tvalid = filter_i_out_tvalid & filter_q_out_tvalid;
	assign data_out_tlast = filter_i_out_tlast & filter_q_out_tlast;



	// instantiation of axi bus interface control
	apsk_modulator_control # ( 
		.BITS_PER_SYMBOL_WIDTH(BITS_PER_SYMBOL_WIDTH),
		.C_S_AXI_DATA_WIDTH(CONTROL_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(CONTROL_ADDR_WIDTH)
	) apsk_modulator_control_inst (
		.bits_per_symbol(bits_per_symbol),
		.offset_symbol_enable(offset_symbol_enable),
		.s_axi_aclk(control_aclk),
		.s_axi_aresetn(control_aresetn),
		.s_axi_awaddr(control_awaddr),
		.s_axi_awprot(control_awprot),
		.s_axi_awvalid(control_awvalid),
		.s_axi_awready(control_awready),
		.s_axi_wdata(control_wdata),
		.s_axi_wstrb(control_wstrb),
		.s_axi_wvalid(control_wvalid),
		.s_axi_wready(control_wready),
		.s_axi_bresp(control_bresp),
		.s_axi_bvalid(control_bvalid),
		.s_axi_bready(control_bready),
		.s_axi_araddr(control_araddr),
		.s_axi_arprot(control_arprot),
		.s_axi_arvalid(control_arvalid),
		.s_axi_arready(control_arready),
		.s_axi_rdata(control_rdata),
		.s_axi_rresp(control_rresp),
		.s_axi_rvalid(control_rvalid),
		.s_axi_rready(control_rready)
	);



	// used to create the GTKwave dump file
	`ifdef COCOTB_SIM
			initial begin
			$dumpfile ("waveform.vcd");
			$dumpvars (0, apsk_modulator);
			#1;
		end
	`endif

	endmodule
