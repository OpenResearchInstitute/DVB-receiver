`timescale 1 ns / 1 ps

module lookup_table #
(
	// signal width definitions
	parameter integer TDATA_WIDTH = 32,
	parameter integer ADDRESS_WIDTH = 8
)
(
	// input data AXI bus
	input wire  							data_in_aclk,
	input wire  							data_in_aresetn,
	output wire								data_in_tready,
	input wire [ADDRESS_WIDTH-1:0] 			data_in_tdata,
	input wire  							data_in_tlast,
	input wire  							data_in_tvalid,

	// output data AXI bus
	output wire  							data_out_aclk,
	output wire  							data_out_aresetn,
	input wire								data_out_tready,
	output wire [TDATA_WIDTH-1:0] 			data_out_tdata,
	output reg  							data_out_tlast,
	output reg  							data_out_tvalid,

	// load new data into lookup table AXI bus
	input wire  							data_load_aclk,
	input wire  							data_load_aresetn,
	output wire								data_load_tready,
	input wire [TDATA_WIDTH-1:0] 			data_load_tdata,
	input wire 								data_load_tlast,
	input wire 								data_load_tvalid
);
	
	// RAM loading control signals
	wire 					data_load;
	reg [ADDRESS_WIDTH-1:0]	data_load_address;


	// RAM output signals
	wire [15:0] 	DOADO;				// port A data output
	wire [15:0] 	DOBDO;				// port B data output
	wire [1:0] 		DOPADOP;			// port A parity output
	wire [1:0] 		DOPBDOP;			// port B parity output

	// RAM input signals
	wire 			CLKARDCLK;			// port A clock input
	wire 			CLKBWRCLK;			// port B clock input
	wire 			ENARDEN;			// port A write enable
	wire 			ENBWREN;			// port B write enable
	wire 			REGCEAREGCE;		// port A output register clock enable
	wire 			REGCEB;				// port B output register clock enable
	wire 			RSTRAMARSTRAM;		// Synchronous output register reset
	wire 			RSTRAMB;			// Synchronous output register reset
	wire 			RSTREGARSTREG;		// Synchronous output register reset
	wire 			RSTREGB;			// Synchronous output register reset
	wire [13:0] 	ADDRARDADDR;		// port A address bus
	wire [13:0] 	ADDRBWRADDR;		// port B address bus
	wire [15:0] 	DIADI;				// port A data input
	wire [15:0] 	DIBDI;				// port B data input
	wire [1:0] 		DIPADIP;			// port A parity inputs
	wire [1:0] 		DIPBDIP;			// port B parity inputs
	wire [1:0] 		WEA;				// port A write enable
	wire [3:0] 		WEBWE;				// port B write enable

	



	// RAM parameters
	localparam integer DOA_REG = 0;
	localparam integer DOB_REG = 0;
	localparam [17:0] INIT_A = 18'h1;
	localparam [17:0] INIT_B = 18'h2;
	localparam INIT_FILE = "NONE";
	localparam IS_CLKARDCLK_INVERTED = 1'b0;
	localparam IS_CLKBWRCLK_INVERTED = 1'b0;
	localparam IS_ENARDEN_INVERTED = 1'b0;
	localparam IS_ENBWREN_INVERTED = 1'b0;
	localparam IS_RSTRAMARSTRAM_INVERTED = 1'b0;
	localparam IS_RSTRAMB_INVERTED = 1'b0;
	localparam IS_RSTREGARSTREG_INVERTED = 1'b0;
	localparam IS_RSTREGB_INVERTED = 1'b0;
	localparam RAM_MODE = "SDP";
	localparam RDADDR_COLLISION_HWCONFIG = "DELAYED_WRITE";
	localparam integer READ_WIDTH_A = 36;
	localparam integer READ_WIDTH_B = 36;
	localparam RSTREG_PRIORITY_A = "RSTREG";
	localparam RSTREG_PRIORITY_B = "RSTREG";
	localparam SIM_COLLISION_CHECK = "ALL";
	localparam SIM_DEVICE = "7SERIES";
	localparam [17:0] SRVAL_A = 18'h0;
	localparam [17:0] SRVAL_B = 18'h0;
	localparam WRITE_MODE_A = "WRITE_FIRST";
	localparam WRITE_MODE_B = "WRITE_FIRST";
	localparam integer WRITE_WIDTH_A = 36;
	localparam integer WRITE_WIDTH_B = 36;



	// pass the ready signal directly through
	assign data_in_tready = data_out_tready;

	// the signals are delayed by a single cycle
	always @(posedge data_in_aclk) begin
		if(!data_in_aresetn) begin
			data_out_tvalid <= 0;
			data_out_tlast <= 0;
		end
		else begin
			data_out_tvalid <= data_in_tvalid;
			data_out_tlast <= data_in_tlast;
		end
	end


	// connect the AXI bus clock to both ports
	assign CLKARDCLK = data_load_aclk;
	assign CLKBRDCLK = data_load_aclk;
	assign CLKBWRCLK = data_load_aclk;

	// input signals that still need terminated
	assign WEBWE = {4{data_load}};
	assign ENBWREN = data_load;
	assign ADDRBWRADDR = {data_load_address, 5'b00000};
	assign DIADI = data_load_tdata[TDATA_WIDTH-1:TDATA_WIDTH/2];
	assign DIBDI = data_load_tdata[TDATA_WIDTH/2-1:0];

	// connect the AXI bus reset signal to the RAM
	assign RSTRAMARSTRAM = !data_load_aresetn;
	assign RSTRAMB = !data_load_aresetn;
	assign RSTREGARSTREG = !data_load_aresetn;
	assign RSTREGB = !data_load_aresetn;

	// map signals to data out ports
	assign ADDRARDADDR = {data_in_tdata, 5'b00000};
	assign REGCEB = !data_load_tvalid;
	assign ENARDEN = !data_load_tvalid;
	assign REGCEAREGCE = !data_load_tvalid;
	assign data_out_tdata = {DOADO, DOBDO};

	// not used in SDP mode
	assign WEA = 0;

	// not currently using parity
	assign DIPADIP = 0;
	assign DIPBDIP = 0;


	// if the AXI bus is ready and has valid data we're loading data
	assign data_load = data_load_tvalid & data_load_tready;

	// generate incrementing address for loading data
	assign data_load_tready = 1;
	always @(posedge data_load_aclk) begin
		if(!data_load_aresetn) begin
			data_load_address <= 0;
		end
		else begin
			if (data_load_tlast) begin
				data_load_address <= 0;
			end
			else if (data_load) begin
				data_load_address <= data_load_address + 1;
			end
		end
	end

	
	// instantiate the RAM module
	RAMB18E1 #(
		.DOA_REG(DOA_REG),
		.DOB_REG(DOB_REG),
		.INIT_A(INIT_A),
		.INIT_B(INIT_B),
		.INIT_FILE(INIT_FILE),
		.IS_CLKARDCLK_INVERTED(IS_CLKARDCLK_INVERTED),
		.IS_CLKBWRCLK_INVERTED(IS_CLKBWRCLK_INVERTED),
		.IS_ENARDEN_INVERTED(IS_ENARDEN_INVERTED),
		.IS_ENBWREN_INVERTED(IS_ENBWREN_INVERTED),
		.IS_RSTRAMARSTRAM_INVERTED(IS_RSTRAMARSTRAM_INVERTED),
		.IS_RSTRAMB_INVERTED(IS_RSTRAMB_INVERTED),
		.IS_RSTREGARSTREG_INVERTED(IS_RSTREGARSTREG_INVERTED),
		.IS_RSTREGB_INVERTED(IS_RSTREGB_INVERTED),
		.RAM_MODE(RAM_MODE),
		.RDADDR_COLLISION_HWCONFIG(RDADDR_COLLISION_HWCONFIG),
		.READ_WIDTH_A(READ_WIDTH_A),
		.READ_WIDTH_B(READ_WIDTH_B),
		.RSTREG_PRIORITY_A(RSTREG_PRIORITY_A),
		.RSTREG_PRIORITY_B(RSTREG_PRIORITY_B),
		.SIM_COLLISION_CHECK(SIM_COLLISION_CHECK),
		.SIM_DEVICE(SIM_DEVICE),
		.SRVAL_A(SRVAL_A),
		.SRVAL_B(SRVAL_B),
		.WRITE_MODE_A(WRITE_MODE_A),
		.WRITE_MODE_B(WRITE_MODE_B),
		.WRITE_WIDTH_A(WRITE_WIDTH_A),
		.WRITE_WIDTH_B(WRITE_WIDTH_B)
	) RAMB18E1_inst (
		.DOADO(DOADO),
		.DOBDO(DOBDO),
		.DOPADOP(DOPADOP),
		.DOPBDOP(DOPBDOP),
		.ADDRARDADDR(ADDRARDADDR),
		.ADDRBWRADDR(ADDRBWRADDR),
		.CLKARDCLK(CLKARDCLK),
		.CLKBWRCLK(CLKBWRCLK),
		.DIADI(DIADI),
		.DIBDI(DIBDI),
		.DIPADIP(DIPADIP),
		.DIPBDIP(DIPBDIP),
		.ENARDEN(ENARDEN),
		.ENBWREN(ENBWREN),
		.REGCEAREGCE(REGCEAREGCE),
		.REGCEB(REGCEB),
		.RSTRAMARSTRAM(RSTRAMARSTRAM),
		.RSTRAMB(RSTRAMB),
		.RSTREGARSTREG(RSTREGARSTREG),
		.RSTREGB(RSTREGB),
		.WEA(WEA),
		.WEBWE(WEBWE)
	);


	// used to create the GTKwave dump file
	`ifdef COCOTB_SIM
			initial begin
			$dumpfile ("waveform.vcd");
			$dumpvars (0, lookup_table);
			#1;
		end
	`endif

endmodule