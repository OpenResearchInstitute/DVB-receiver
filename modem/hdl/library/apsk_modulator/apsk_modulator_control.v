
`timescale 1 ns / 1 ps

	module apsk_modulator_control #
	(
		// user register parameters
		parameter integer BITS_PER_SYMBOL_WIDTH = 4,

		// Width of S_AXI data bus
		parameter integer C_S_AXI_DATA_WIDTH	= 32,
		// Width of S_AXI address bus
		parameter integer C_S_AXI_ADDR_WIDTH	= 4
	)
	(
		// user register ports
		output wire [BITS_PER_SYMBOL_WIDTH-1:0] 	bits_per_symbol,
		output wire									offset_symbol_enable,

		// Global Clock Signal
		input wire  								s_axi_aclk,
		// Global Reset Signal. This Signal is Active LOW
		input wire  								s_axi_aresetn,
		// Write address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1:0] 		s_axi_awaddr,
		// Write channel Protection type. This signal indicates the
			// privilege and security level of the transaction, and whether
			// the transaction is a data access or an instruction access.
		input wire [2:0] 							s_axi_awprot,
		// Write address valid. This signal indicates that the master signaling
			// valid write address and control information.
		input wire  								s_axi_awvalid,
		// Write address ready. This signal indicates that the slave is ready
			// to accept an address and associated control signals.
		output wire  								s_axi_awready,
		// Write data (issued by master, acceped by Slave) 
		input wire [C_S_AXI_DATA_WIDTH-1:0] 		s_axi_wdata,
		// Write strobes. This signal indicates which byte lanes hold
			// valid data. There is one write strobe bit for each eight
			// bits of the write data bus.    
		input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] 	s_axi_wstrb,
		// Write valid. This signal indicates that valid write
			// data and strobes are available.
		input wire  								s_axi_wvalid,
		// Write ready. This signal indicates that the slave
			// can accept the write data.
		output wire  								s_axi_wready,
		// Write response. This signal indicates the status
			// of the write transaction.
		output wire [1:0] 							s_axi_bresp,
		// Write response valid. This signal indicates that the channel
			// is signaling a valid write response.
		output wire  								s_axi_bvalid,
		// Response ready. This signal indicates that the master
			// can accept a write response.
		input wire  								s_axi_bready,
		// Read address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1:0] 		s_axi_araddr,
		// Protection type. This signal indicates the privilege
			// and security level of the transaction, and whether the
			// transaction is a data access or an instruction access.
		input wire [2:0] 							s_axi_arprot,
		// Read address valid. This signal indicates that the channel
			// is signaling valid read address and control information.
		input wire  								s_axi_arvalid,
		// Read address ready. This signal indicates that the slave is
			// ready to accept an address and associated control signals.
		output wire  								s_axi_arready,
		// Read data (issued by slave)
		output wire [C_S_AXI_DATA_WIDTH-1:0] 		s_axi_rdata,
		// Read response. This signal indicates the status of the
			// read transfer.
		output wire [1:0] 							s_axi_rresp,
		// Read valid. This signal indicates that the channel is
			// signaling the required read data.
		output wire  								s_axi_rvalid,
		// Read ready. This signal indicates that the master can
			// accept the read data and response information.
		input wire  								s_axi_rready
	);

	// AXI4LITE signals
	reg [C_S_AXI_ADDR_WIDTH-1:0] 		axi_awaddr;
	reg  								axi_awready;
	reg  								axi_wready;
	reg [1:0] 							axi_bresp;
	reg  								axi_bvalid;
	reg [C_S_AXI_ADDR_WIDTH-1:0] 		axi_araddr;
	reg  								axi_arready;
	reg [C_S_AXI_DATA_WIDTH-1:0] 		axi_rdata;
	reg [1:0] 							axi_rresp;
	reg  								axi_rvalid;

	// Example-specific design signals
	// local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
	// ADDR_LSB is used for addressing 32/64 bit registers/memories
	// ADDR_LSB = 2 for 32 bits (n downto 2)
	// ADDR_LSB = 3 for 64 bits (n downto 3)
	localparam integer ADDR_LSB 			= (C_S_AXI_DATA_WIDTH/32) + 1;
	localparam integer OPT_MEM_ADDR_BITS 	= 1;

	//----------------------------------------------
	//-- Signals for user logic register space example
	//------------------------------------------------
	//-- Number of Slave Registers 4
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg0;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg1;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg2;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg3;
	wire	 						slv_reg_rden;
	wire	 						slv_reg_wren;
	reg [C_S_AXI_DATA_WIDTH-1:0] 	reg_data_out;
	integer	 						byte_index;
	reg	 							aw_en;

	// I/O Connections assignments
	assign s_axi_awready	= axi_awready;
	assign s_axi_wready		= axi_wready;
	assign s_axi_bresp		= axi_bresp;
	assign s_axi_bvalid		= axi_bvalid;
	assign s_axi_arready 	= axi_arready;
	assign s_axi_rdata		= axi_rdata;
	assign s_axi_rresp		= axi_rresp;
	assign s_axi_rvalid		= axi_rvalid;


	// Implement axi_awready generation
	// axi_awready is asserted for one S_AXI_ACLK clock cycle when both
	// S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_awready is
	// de-asserted when reset is low.
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 )
	    begin
			axi_awready <= 1'b0;
			aw_en <= 1'b1;
	    end 
		else begin

			if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
				// slave is ready to accept write address when 
				// there is a valid write address and write data
				// on the write address and data bus. This design 
				// expects no outstanding transactions. 
				axi_awready <= 1'b1;
				aw_en <= 1'b0;
	        end
	        else if (s_axi_bready && axi_bvalid) begin
				aw_en <= 1'b1;
				axi_awready <= 1'b0;
			end
			else begin
				axi_awready <= 1'b0;
			end
		end
	end  

	// Implement axi_awaddr latching
	// This process is used to latch the address when both 
	// S_AXI_AWVALID and S_AXI_WVALID are valid. 
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_awaddr <= 0;
		end 
		else begin    
			if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
				// Write Address latching 
				axi_awaddr <= s_axi_awaddr;
			end
		end 
	end       

	// Implement axi_wready generation
	// axi_wready is asserted for one S_AXI_ACLK clock cycle when both
	// S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_wready is 
	// de-asserted when reset is low. 
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_wready <= 1'b0;
		end
		else begin    
			if (~axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en ) begin
				// slave is ready to accept write data when 
				// there is a valid write address and write data
				// on the write address and data bus. This design 
				// expects no outstanding transactions. 
				axi_wready <= 1'b1;
			end
			else begin
				axi_wready <= 1'b0;
			end
		end
	end      

	// Implement memory mapped register select and write logic generation
	// The write data is accepted and written to memory mapped registers when
	// axi_awready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted. Write strobes are used to
	// select byte enables of slave registers while writing.
	// These registers are cleared when reset (active low) is applied.
	// Slave register write enable is asserted when valid address and data are available
	// and the slave is ready to accept the write address and write data.
	assign slv_reg_wren = axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid;

	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			slv_reg0 <= 0;
			slv_reg1 <= 0;
			slv_reg2 <= 0;
			slv_reg3 <= 0;
		end 
		else begin
			if (slv_reg_wren) begin
				case ( axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
					2'h0:
						for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
							if ( s_axi_wstrb[byte_index] == 1 ) begin
								// Respective byte enables are asserted as per write strobes 
								// Slave register 0
								slv_reg0[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
							end  
					2'h1:
						for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
							if ( s_axi_wstrb[byte_index] == 1 ) begin
								// Respective byte enables are asserted as per write strobes 
								// Slave register 1
								slv_reg1[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
							end  
					2'h2:
						for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
							if ( s_axi_wstrb[byte_index] == 1 ) begin
								// Respective byte enables are asserted as per write strobes 
								// Slave register 2
								slv_reg2[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
							end  
					2'h3:
						for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
							if ( s_axi_wstrb[byte_index] == 1 ) begin
								// Respective byte enables are asserted as per write strobes 
								// Slave register 3
								slv_reg3[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
							end  
					default : begin
						slv_reg0 <= slv_reg0;
						slv_reg1 <= slv_reg1;
						slv_reg2 <= slv_reg2;
						slv_reg3 <= slv_reg3;
					end
				endcase
			end
		end
	end    

	// Implement write response logic generation
	// The write response and response valid signals are asserted by the slave 
	// when axi_wready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted.  
	// This marks the acceptance of address and indicates the status of 
	// write transaction.
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_bvalid  <= 0;
			axi_bresp   <= 2'b0;
		end 
		else begin    
			if (axi_awready && s_axi_awvalid && ~axi_bvalid && axi_wready && s_axi_wvalid) begin
				// indicates a valid write response is available
				axi_bvalid <= 1'b1;
				axi_bresp  <= 2'b0; // 'OKAY' response 
			end                   // work error responses in future
			else begin
				if (s_axi_bready && axi_bvalid) begin
					//check if bready is asserted while bvalid is high) 
					//(there is a possibility that bready is always asserted high)   
					axi_bvalid <= 1'b0; 
				end  
			end
		end
	end   

	// Implement axi_arready generation
	// axi_arready is asserted for one S_AXI_ACLK clock cycle when
	// S_AXI_ARVALID is asserted. axi_awready is 
	// de-asserted when reset (active low) is asserted. 
	// The read address is also latched when S_AXI_ARVALID is 
	// asserted. axi_araddr is reset to zero on reset assertion.
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_arready <= 1'b0;
			axi_araddr  <= 32'b0;
		end 
		else begin    
			if (~axi_arready && s_axi_arvalid) begin
				// indicates that the slave has acceped the valid read address
				axi_arready <= 1'b1;
				// Read address latching
				axi_araddr  <= s_axi_araddr;
			end
			else begin
				axi_arready <= 1'b0;
			end
		end 
	end       

	// Implement axi_arvalid generation
	// axi_rvalid is asserted for one S_AXI_ACLK clock cycle when both 
	// S_AXI_ARVALID and axi_arready are asserted. The slave registers 
	// data are available on the axi_rdata bus at this instance. The 
	// assertion of axi_rvalid marks the validity of read data on the 
	// bus and axi_rresp indicates the status of read transaction.axi_rvalid 
	// is deasserted on reset (active low). axi_rresp and axi_rdata are 
	// cleared to zero on reset (active low).  
	always @( posedge s_axi_aclk ) begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_rvalid <= 0;
			axi_rresp  <= 0;
		end 
		else begin    
			if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
				// Valid read data is available at the read data bus
				axi_rvalid <= 1'b1;
				axi_rresp  <= 2'b0; // 'OKAY' response
			end   
			else if (axi_rvalid && s_axi_rready) begin
				// Read data is accepted by the master
				axi_rvalid <= 1'b0;
			end
		end
	end

	// Implement memory mapped register select and read logic generation
	// Slave register read enable is asserted when valid address is available
	// and the slave is ready to accept the read address.
	assign slv_reg_rden = axi_arready & s_axi_arvalid & ~axi_rvalid;
	always @(*)
	begin
		// Address decoding for reading registers
		case ( axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
			2'h0  :reg_data_out <= slv_reg0;
			2'h1  :reg_data_out <= slv_reg1;
			2'h2  :reg_data_out <= slv_reg2;
			2'h3  :reg_data_out <= slv_reg3;
			default:reg_data_out <= 0;
		endcase
	end

	// Output register or memory read data
	always @( posedge s_axi_aclk )
	begin
		if ( s_axi_aresetn == 1'b0 ) begin
			axi_rdata  <= 0;
		end
		else begin
			// When there is a valid read address (S_AXI_ARVALID) with
			// acceptance of read address by the slave (axi_arready),
			// output the read dada 
			if (slv_reg_rden) begin
				axi_rdata <= reg_data_out;     // register read data
			end
		end
	end

	// Add user logic here
    assign bits_per_symbol = reg_data_out[BITS_PER_SYMBOL_WIDTH-1:0];
	assign offset_symbol_enable = reg_data_out[BITS_PER_SYMBOL_WIDTH];

	endmodule