`timescale 1 ns / 1 ps

module lookup_table_behavioural #
(
    // signal width definitions
    parameter integer TDATA_WIDTH = 32,
    parameter integer ADDRESS_WIDTH = 8
)
(
    // input data AXI bus
    input wire                              data_in_aclk,
    input wire                              data_in_aresetn,
    output wire                             data_in_tready,
    input wire [ADDRESS_WIDTH-1:0]          data_in_tdata,
    input wire                              data_in_tlast,
    input wire                              data_in_tvalid,

    // output data AXI bus
    output wire                             data_out_aclk,
    output wire                             data_out_aresetn,
    input wire                              data_out_tready,
    output reg [TDATA_WIDTH-1:0]            data_out_tdata,
    output reg                              data_out_tlast,
    output reg                              data_out_tvalid,

    // load new data into lookup table AXI bus
    input wire                              data_load_aclk,
    input wire                              data_load_aresetn,
    output wire                             data_load_tready,
    input wire [TDATA_WIDTH-1:0]            data_load_tdata,
    input wire                              data_load_tlast,
    input wire                              data_load_tvalid
);


    reg [TDATA_WIDTH-1:0]       register_space[2**ADDRESS_WIDTH-1:0];


    // load the data into the register space structure
    integer i;
    always @(posedge data_load_aclk) begin
        if(!data_load_aresetn) begin
            for (i = 0; i < ADDRESS_WIDTH; i=i+1) begin
                register_space[i] <= 0;
            end
        end
        else begin
            if (data_load_tvalid & data_load_tready) begin
                register_space[data_load_address] <= data_load_tdata;
            end
        end
    end


    // index out the data
    always @(posedge data_out_aclk) begin
        if(!data_out_aresetn) begin
            data_out_tdata <= 0;
        end
        else begin
            data_out_tdata <= register_space[data_in_tdata];
        end
    end


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


    // if the AXI bus is ready and has valid data we're loading data
    assign data_load = data_load_tvalid & data_load_tready;

    // generate incrementing address for loading data
    reg [ADDRESS_WIDTH-1:0] data_load_address;
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

    

    // used to create the GTKwave dump file
    `ifdef COCOTB_SIM
            initial begin
            $dumpfile ("waveform.vcd");
            $dumpvars (0, lookup_table_behavioural);
            #1;
        end
    `endif

endmodule