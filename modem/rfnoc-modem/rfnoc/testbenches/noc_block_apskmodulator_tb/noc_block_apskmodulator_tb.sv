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

`timescale 1ns/1ps
`define NS_PER_TICK 1
`define NUM_TEST_CASES 5

`include "sim_exec_report.vh"
`include "sim_clks_rsts.vh"
`include "sim_rfnoc_lib.svh"

module noc_block_apskmodulator_tb();
  `TEST_BENCH_INIT("noc_block_apskmodulator",`NUM_TEST_CASES,`NS_PER_TICK);
  localparam BUS_CLK_PERIOD = $ceil(1e9/166.67e6);
  localparam CE_CLK_PERIOD  = $ceil(1e9/200e6);
  localparam NUM_CE         = 1;  // Number of Computation Engines / User RFNoC blocks to simulate
  localparam NUM_STREAMS    = 1;  // Number of test bench streams
  `RFNOC_SIM_INIT(NUM_CE, NUM_STREAMS, BUS_CLK_PERIOD, CE_CLK_PERIOD);
  `RFNOC_ADD_BLOCK(noc_block_apskmodulator, 0);

  // coefficient data for the test
  localparam NUM_COEFFS  = 40;
  localparam COEFF_WIDTH = 16;
  localparam [COEFF_WIDTH*NUM_COEFFS-1:0] COEFFS_VEC =
    {16'd65531, 16'd48, 16'd40, 16'd65505, 16'd65454, 16'd65505, 16'd87, 16'd134, 16'd24, 16'd65402, 
     16'd65414, 16'd126, 16'd346, 16'd126, 16'd64924, 16'd64255, 16'd64670, 16'd1281, 16'd4727, 16'd7961, 
     16'd9286, 16'd7961, 16'd4727, 16'd1281, 16'd64670, 16'd64255, 16'd64924, 16'd126, 16'd346, 16'd126, 
     16'd65414, 16'd65402, 16'd24, 16'd134, 16'd87, 16'd65505, 16'd65454, 16'd65505, 16'd40, 16'd48};

  // constellation data for the test
  localparam NUM_CONSTELLATION_POINT  = 8;
  localparam CONSTELLATION_POINT_WIDTH = 32;
  localparam [CONSTELLATION_POINT_WIDTH*NUM_CONSTELLATION_POINT-1:0] CONSTELLATION_POINT_VEC =
    {32'd759246145, 32'd16383, 32'd49153, 32'd3535786687, 32'd1073676288, 
     32'd3535744321, 32'd759288511, 32'd3221356544};

  // input test data
  localparam NUM_INPUT_SAMPLES = 8;
  localparam INPUT_SAMPLES_WIDTH = 32;
  localparam [INPUT_SAMPLES_WIDTH*NUM_INPUT_SAMPLES-1:0] INPUT_SAMPLES_VEC =
    {32'd3882183212, 32'd3074848329, 32'd871663141, 32'd3761980262, 32'd607163576, 32'd1851188060, 32'd2833901076, 32'd3722575619};

  // output test data
  localparam NUM_OUTPUT_SAMPLES = 376;
  localparam OUTPUT_SAMPLES_WIDTH = 16;
  localparam [OUTPUT_SAMPLES_WIDTH*NUM_OUTPUT_SAMPLES-1:0] OUTPUT_I_SAMPLES_VEC =
    {16'sd0, 16'sd0, 16'sd0, 16'sd0, -16'sd11, 16'sd93, 16'sd79, -16'sd62, -16'sd175, 16'sd34, 16'sd252, 16'sd204, -16'sd127, 
    -16'sd235, 16'sd8, 16'sd456, 16'sd586, -16'sd175, -16'sd1377, -16'sd1987, -16'sd816, 16'sd2477, 16'sd7714, 16'sd13460, 16'sd17741, 
    16'sd19269, 16'sd17835, 16'sd15238, 16'sd14158, 16'sd15901, 16'sd19462, 16'sd21430, 16'sd18558, 16'sd10398, -16'sd257, -16'sd9687, 
    -16'sd16222, -16'sd21013, -16'sd25665, -16'sd29098, -16'sd27162, -16'sd16162, 16'sd2463, 16'sd21016, 16'sd29705, 16'sd23584, 
    16'sd6157, -16'sd12383, -16'sd22263, -16'sd20279, -16'sd10960, -16'sd1672, 16'sd2812, 16'sd2605, 16'sd552, -16'sd643, -16'sd405, 
    16'sd309, 16'sd456, -16'sd19, -16'sd436, -16'sd515, 16'sd31, 16'sd527, 16'sd385, -16'sd240, -16'sd581, -16'sd51, 16'sd799, 16'sd677, 
    -16'sd1077, -16'sd3152, -16'sd2653, 16'sd2616, 16'sd11201, 16'sd18478, 16'sd19550, 16'sd12709, 16'sd1074, -16'sd10240, -16'sd17699, 
    -16'sd21146, -16'sd21906, -16'sd20797, -16'sd17103, -16'sd10084, -16'sd255, 16'sd9840, 16'sd17375, 16'sd21322, 16'sd22005, 16'sd20437, 
    16'sd16718, 16'sd10302, 16'sd754, -16'sd10104, -16'sd18654, -16'sd21866, -16'sd19632, -16'sd15210, -16'sd13035, -16'sd15241, -16'sd20052, 
    -16'sd22813, -16'sd19363, -16'sd9378, 16'sd4218, 16'sd16522, 16'sd24299, 16'sd27613, 16'sd28798, 16'sd28900, 16'sd25883, 16'sd16627, 
    16'sd1051, -16'sd15042, -16'sd22759, -16'sd16851, -16'sd255, 16'sd16607, 16'sd23037, 16'sd15334, -16'sd1142, -16'sd17186, -16'sd26479, 
    -16'sd28594, -16'sd27309, -16'sd25719, -16'sd23681, -16'sd19227, -16'sd11805, -16'sd3617, 16'sd1916, 16'sd3285, 16'sd1652, -16'sd48, 
    -16'sd292, -16'sd79, -16'sd1233, -16'sd3385, -16'sd3163, 16'sd3158, 16'sd14552, 16'sd25268, 16'sd28106, 16'sd19729, 16'sd3813, 
    -16'sd11833, -16'sd20298, -16'sd19814, -16'sd12865, -16'sd4360, 16'sd2134, 16'sd5814, 16'sd8366, 16'sd11487, 16'sd15635, 16'sd20182, 
    16'sd23550, 16'sd24707, 16'sd23321, 16'sd19666, 16'sd13908, 16'sd6713, -16'sd652, -16'sd6580, -16'sd10481, -16'sd12774, -16'sd14813, 
    -16'sd17908, -16'sd21183, -16'sd22286, -16'sd18575, -16'sd9128, 16'sd4076, 16'sd16919, 16'sd25359, 16'sd27533, 16'sd24625, 16'sd19550, 
    16'sd14872, 16'sd11751, 16'sd8975, 16'sd5465, 16'sd805, -16'sd4709, -16'sd10733, -16'sd16057, -16'sd18481, -16'sd15961, -16'sd8499, 
    16'sd172, 16'sd4306, 16'sd362, -16'sd9863, -16'sd19578, -16'sd21390, -16'sd12550, 16'sd3115, 16'sd18127, 16'sd26476, 16'sd27324, 
    16'sd23587, 16'sd19320, 16'sd16125, 16'sd12457, 16'sd6982, 16'sd632, -16'sd2939, -16'sd481, 16'sd8349, 16'sd18422, 16'sd22124, 
    16'sd14900, -16'sd1128, -16'sd17211, -16'sd23485, -16'sd15819, 16'sd1315, 16'sd17259, 16'sd22416, 16'sd14322, -16'sd1485, -16'sd16454, 
    -16'sd25316, -16'sd28509, -16'sd29164, -16'sd28628, -16'sd25030, -16'sd16006, -16'sd2358, 16'sd11232, 16'sd19281, 16'sd19377, 16'sd13548, 
    16'sd5871, -16'sd1014, -16'sd7286, -16'sd13477, -16'sd19391, -16'sd23624, -16'sd24749, -16'sd23332, -16'sd20491, -16'sd16786, -16'sd12017, 
    -16'sd6035, -16'sd541, 16'sd1689, -16'sd1037, -16'sd6923, -16'sd12854, -16'sd16774, -16'sd18940, -16'sd20437, -16'sd20945, -16'sd18317,
    -16'sd10673, 16'sd1006, 16'sd12542, 16'sd19094, 16'sd18215, 16'sd12224, 16'sd4890, -16'sd1213, -16'sd5726, -16'sd9573, -16'sd13250, 
    -16'sd16292, -16'sd18260, -16'sd19011, -16'sd18858, -16'sd17427, -16'sd13721, -16'sd6920, 16'sd697, 16'sd4150, -16'sd893, -16'sd13086, 
    -16'sd25484, -16'sd29368, -16'sd20579, -16'sd2673, 16'sd14790, 16'sd22411, 16'sd16919, 16'sd2704, -16'sd11847, -16'sd20205, -16'sd21166, 
    -16'sd18260, -16'sd15842, -16'sd15674, -16'sd16412, -16'sd16227, -16'sd15124, -16'sd14943, -16'sd17347, -16'sd21495, -16'sd24735, 
    -16'sd24605, -16'sd20403, -16'sd13557, -16'sd6129, 16'sd544, 16'sd6268, 16'sd10543, 16'sd13333, 16'sd15232, 16'sd17171, 16'sd20018, 
    16'sd22127, 16'sd19902, 16'sd10654, -16'sd3708, -16'sd16794, -16'sd21237, -16'sd14036, 16'sd1355, 16'sd17027, 16'sd26357, 16'sd28106, 
    16'sd25370, 16'sd21486, 16'sd16851, 16'sd9624, -16'sd989, -16'sd12352, -16'sd19516, -16'sd18881, -16'sd11490, -16'sd2180, 16'sd3651, 
    16'sd3521, -16'sd669, -16'sd4275, -16'sd3180, 16'sd3716, 16'sd13837, 16'sd22578, 16'sd25898, 16'sd22442, 16'sd13724, 16'sd3804, -16'sd2937, 
    -16'sd4090, -16'sd833, 16'sd3078, 16'sd3407, -16'sd1959, -16'sd11050, -16'sd18906, -16'sd20284, -16'sd13012, 16'sd243, 16'sd13302, 16'sd20117, 
    16'sd18436, 16'sd10815, 16'sd2449, -16'sd2429, -16'sd2820, -16'sd983, 16'sd518, 16'sd643, -16'sd17, -16'sd419, -16'sd206, 16'sd209, 16'sd328, 
    16'sd93, -16'sd158, -16'sd164, -16'sd62, 16'sd79, 16'sd93};
  localparam [OUTPUT_SAMPLES_WIDTH*NUM_OUTPUT_SAMPLES-1:0] OUTPUT_Q_SAMPLES_VEC =
    {-16'sd14, 16'sd133, 16'sd110, -16'sd87, -16'sd223, -16'sd184, 16'sd164, 16'sd439, 16'sd221, -16'sd223, -16'sd442, 16'sd25, 16'sd776, 16'sd464,
    -16'sd1397, -16'sd3555, -16'sd2948, 16'sd3265, 16'sd14285, 16'sd25056, 16'sd28540, 16'sd20457, 16'sd3115, -16'sd14881, -16'sd23454, -16'sd17546,
    -16'sd754, 16'sd16567, 16'sd23769, 16'sd16471, -16'sd428, -16'sd16531, -16'sd22535, -16'sd15907, -16'sd1023, 16'sd13642, 16'sd20933, 16'sd18915, 
    16'sd10506, 16'sd1522, -16'sd3197, -16'sd2588, 16'sd816, 16'sd3039, 16'sd1522, -16'sd3161, -16'sd8558, -16'sd12550, -16'sd15283, -16'sd18475,
    -16'sd23811, -16'sd28991, -16'sd28367, -16'sd17120, 16'sd3056, 16'sd23060, 16'sd31366, 16'sd22107, 16'sd65, -16'sd21858, -16'sd31040,
    -16'sd23250, -16'sd3898, 16'sd16253, 16'sd28486, 16'sd30652, 16'sd26278, 16'sd21237, 16'sd19816, 16'sd22728, 16'sd27270, 16'sd29232, 16'sd25484,
    16'sd15578, 16'sd2786, -16'sd8975, -16'sd17137, -16'sd21384, -16'sd22626, -16'sd21296, -16'sd16979, -16'sd9687, 16'sd2, 16'sd9778, 16'sd17188,
    16'sd21115, 16'sd21999, 16'sd20698, 16'sd17061, 16'sd10291, 16'sd252, -16'sd10625, -16'sd18549, -16'sd21033, -16'sd19207, -16'sd16261,
    -16'sd14940, -16'sd15533, -16'sd16570, -16'sd16701, -16'sd16128, -16'sd15972, -16'sd16973, -16'sd18271, -16'sd17903, -16'sd14305, -16'sd8170,
    -16'sd1873, 16'sd1899, 16'sd2092, -16'sd255, -16'sd2336, -16'sd1607, 16'sd2222, 16'sd8125, 16'sd13713, 16'sd17239, 16'sd18271, 16'sd18337,
    16'sd18198, 16'sd17191, 16'sd14056, 16'sd8380, 16'sd1896, -16'sd2324, -16'sd2381, 16'sd853, 16'sd3637, 16'sd2273, -16'sd3753, -16'sd12000,
    -16'sd18972, -16'sd23182, -16'sd25560, -16'sd27993, -16'sd29617, -16'sd26663, -16'sd15819, 16'sd1735, 16'sd19369, 16'sd28724, 16'sd25776,
    16'sd13732, 16'sd1173, -16'sd4323, -16'sd1040, 16'sd6461, 16'sd12811, 16'sd16366, 16'sd19408, 16'sd23723, 16'sd27593, 16'sd26516, 16'sd17551,
    16'sd2738, -16'sd11737, -16'sd19746, -16'sd19400, -16'sd13148, -16'sd5454, 16'sd1040, 16'sd6903, 16'sd13063, 16'sd19493, 16'sd24313, 16'sd25526,
    16'sd22507, 16'sd17478, 16'sd13968, 16'sd14084, 16'sd17075, 16'sd19774, 16'sd18935, 16'sd13880, 16'sd6188, -16'sd223, -16'sd2001, 16'sd1071,
    16'sd6926, 16'sd12411, 16'sd16207, 16'sd19451, 16'sd23624, 16'sd27596, 16'sd27148, 16'sd18166, 16'sd1380, -16'sd15864, -16'sd23539, -16'sd15825,
    16'sd3189, 16'sd22394, 16'sd30533, 16'sd23468, 16'sd5551, -16'sd12958, -16'sd22660, -16'sd20125, -16'sd9505, 16'sd1196, 16'sd5026, 16'sd121,
    -16'sd10381, -16'sd19567, -16'sd20956, -16'sd12216, 16'sd3036, 16'sd18036, 16'sd26785, 16'sd27329, 16'sd22569, 16'sd17220, 16'sd14577, 16'sd15051,
    16'sd16233, 16'sd16426, 16'sd15924, 16'sd15847, 16'sd17265, 16'sd18816, 16'sd18084, 16'sd13903, 16'sd7677, 16'sd2239, -16'sd592, -16'sd1598,
    -16'sd2146, -16'sd2806, -16'sd1990, 16'sd2866, 16'sd11158, 16'sd18997, 16'sd20681, 16'sd12615, -16'sd3260, -16'sd19777, -16'sd28676, -16'sd25864,
    -16'sd13920, -16'sd564, 16'sd5990, 16'sd1905, -16'sd9732, -16'sd20675, -16'sd22595, -16'sd12202, 16'sd5854, 16'sd22274, 16'sd28599, 16'sd21957,
    16'sd6622, -16'sd9639, -16'sd20247, -16'sd22992, -16'sd19930, -16'sd15320, -16'sd13129, -16'sd15048, -16'sd19797, -16'sd23338, -16'sd20769,
    -16'sd9641, 16'sd7280, 16'sd22527, 16'sd28186, 16'sd21421, 16'sd6293, -16'sd9250, -16'sd19085, -16'sd22419, -16'sd22340, -16'sd21237, -16'sd18036,
    -16'sd9721, 16'sd4210, 16'sd19255, 16'sd28089, 16'sd25688, 16'sd14149, 16'sd972, -16'sd5647, -16'sd2131, 16'sd8731, 16'sd19332, 16'sd22107,
    16'sd13925, -16'sd1698, -16'sd16539, -16'sd22325, -16'sd15587, 16'sd0, 16'sd15584, 16'sd22323, 16'sd16344, 16'sd1533, -16'sd13803, -16'sd21775,
    -16'sd18980, -16'sd8887, 16'sd1443, 16'sd5151, -16'sd683, -16'sd13321, -16'sd25484, -16'sd29198, -16'sd20347, -16'sd2523, 16'sd15195, 16'sd23111,
    16'sd16976, 16'sd666, -16'sd15952, -16'sd23006, -16'sd16721, -16'sd1145, 16'sd14535, 16'sd21963, 16'sd18334, 16'sd8377, -16'sd473, -16'sd3027,
    16'sd822, 16'sd7424, 16'sd12975, 16'sd16295, 16'sd18606, 16'sd20806, 16'sd21600, 16'sd18345, 16'sd9508, -16'sd3189, -16'sd15808, -16'sd24786,
    -16'sd28772, -16'sd28704, -16'sd26204, -16'sd22368, -16'sd17838, -16'sd13029, -16'sd7977, -16'sd1868, 16'sd6546, 16'sd17049, 16'sd26238,
    16'sd28211, 16'sd18657, -16'sd802, -16'sd21194, -16'sd31043, -16'sd24177, -16'sd4794, 16'sd15204, 16'sd24018, 16'sd17466, 16'sd1272, -16'sd14265,
    -16'sd21265, -16'sd18504, -16'sd10112, -16'sd2024, 16'sd2129, 16'sd2350, 16'sd847, -16'sd297, -16'sd413, 16'sd102, 16'sd303, 16'sd68, -16'sd212,
    -16'sd331, -16'sd96, 16'sd155, 16'sd161, 16'sd59, -16'sd82, -16'sd96};
  localparam SPP = 16; // Samples per packet

  /********************************************************
  ** Verification
  ********************************************************/
  initial begin : tb_main
    string s;
    logic [31:0] random_word;
    logic [63:0] readback;

    logic [COEFF_WIDTH-1:0] coeffs[0:NUM_COEFFS-1];

    logic [CONSTELLATION_POINT_WIDTH-1:0] constellation_points[0:NUM_CONSTELLATION_POINT-1];

    /********************************************************
    ** Test 1 -- Reset
    ********************************************************/
    `TEST_CASE_START("Wait for Reset");
    while (bus_rst) @(posedge bus_clk);
    while (ce_rst) @(posedge ce_clk);
    `TEST_CASE_DONE(~bus_rst & ~ce_rst);

    /********************************************************
    ** Test 2 -- Check for correct NoC IDs
    ********************************************************/
    `TEST_CASE_START("Check NoC ID");
    // Read NOC IDs
    tb_streamer.read_reg(sid_noc_block_apskmodulator, RB_NOC_ID, readback);
    $display("Read apskmodulator NOC ID: %16x", readback);
    `ASSERT_ERROR(readback == noc_block_apskmodulator.NOC_ID, "Incorrect NOC ID");
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 3 -- Connect RFNoC blocks
    ********************************************************/
    `TEST_CASE_START("Connect RFNoC blocks");
    `RFNOC_CONNECT(noc_block_tb,noc_block_apskmodulator,SC16,SPP);
    `RFNOC_CONNECT(noc_block_apskmodulator,noc_block_tb,SC16,SPP);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 4 -- Write / readback user registers
    ********************************************************/
    `TEST_CASE_START("Write / readback user registers");
    random_word = $urandom_range(15,0);
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_BITS_PER_SYMBOL, random_word);
    tb_streamer.read_user_reg(sid_noc_block_apskmodulator, 0, readback);
    $sformat(s, "User register 0 incorrect readback! Expected: %0d, Actual %0d", readback[3:0], random_word);
    `ASSERT_ERROR(readback[3:0] == random_word, s);
    random_word = $urandom_range(1,0);
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_OFFSET_SYMBOL_ENABLE, random_word);
    tb_streamer.read_user_reg(sid_noc_block_apskmodulator, 1, readback);
    $sformat(s, "User register 1 incorrect readback! Expected: %0d, Actual %0d", readback[0:0], random_word);
    `ASSERT_ERROR(readback[0:0] == random_word, s);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 5 -- Load new coefficients
    ********************************************************/
    `TEST_CASE_START("Load new coefficients");
    // for (int i = 0; i < NUM_COEFFS-1; i++) begin
    for (int i = NUM_COEFFS-1; i > 0 ; i--) begin
      tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_COEFFS,
        COEFFS_VEC[COEFF_WIDTH*i +: COEFF_WIDTH]);
    end
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_COEFFS_TLAST,
        COEFFS_VEC[COEFF_WIDTH*0 +: COEFF_WIDTH]);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 6 -- Load new constellation map
    ********************************************************/
    `TEST_CASE_START("Load new constellation map");
    // for (int i = 0; i < NUM_CONSTELLATION_POINT-1; i++) begin
    for (int i = NUM_CONSTELLATION_POINT-1; i > 0 ; i--) begin
      tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_CONSTS,
        CONSTELLATION_POINT_VEC[CONSTELLATION_POINT_WIDTH*i +: CONSTELLATION_POINT_WIDTH]);
    end
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_CONSTS_TLAST,
        CONSTELLATION_POINT_VEC[CONSTELLATION_POINT_WIDTH*0 +: CONSTELLATION_POINT_WIDTH]);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 7 -- Impulse Response with default coefficients
    ********************************************************/
    // Sending an impulse will readback the FIR filter coefficients
    `TEST_CASE_START("Test modulation");

    // write the settings
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_BITS_PER_SYMBOL, 3);
    tb_streamer.write_user_reg(sid_noc_block_apskmodulator, noc_block_apskmodulator.SR_OFFSET_SYMBOL_ENABLE, 0);

    // Send and check impulse
    fork
      begin
        $display("Send data");
        // tb_streamer.push_word(INPUT_SAMPLES_VEC, 0);
        for (int i = NUM_INPUT_SAMPLES-1; i >= 0; i--) begin
          tb_streamer.push_word(INPUT_SAMPLES_VEC[INPUT_SAMPLES_WIDTH*i +: INPUT_SAMPLES_WIDTH], (i == 0) /* Assert tlast on last word */);
        end
        // // Send impulse
        // for (int i = 1; i < num_coeffs; i++) begin
        //   tb_streamer.push_word(0, (i == num_coeffs-1) /* Assert tlast on last word */);
        // end
        // // Send another two packets with 0s to push out the impulse from the pipeline
        // // Why two? One to push out the data and one to overcome some pipeline registering
        // for (int n = 0; n < 2; n++) begin
        //   for (int i = 0; i < num_coeffs; i++) begin
        //     tb_streamer.push_word(0, (i == num_coeffs-1) /* Assert tlast on last word */);
        //   end
        // end
      end
      begin
        logic [31:0] recv_val;
        logic last;
        logic signed [15:0] i_samp, q_samp, i_expected, q_expected;
        $display("Receive modulated output");
        // // Ignore the first two packets
        // // Data is not useful until the pipeline is flushed
        // for (int n = 0; n < 2; n++) begin
        //   for (int i = 0; i < num_coeffs; i++) begin
        //     tb_streamer.pull_word({i_samp, q_samp}, last);
        //   end
        // end
        for (int i = NUM_OUTPUT_SAMPLES-1; i >= 0; i--) begin
          tb_streamer.pull_word({q_samp, i_samp}, last);
          i_expected = $signed(OUTPUT_I_SAMPLES_VEC[OUTPUT_SAMPLES_WIDTH*i +: OUTPUT_SAMPLES_WIDTH])/2.834630687;
          q_expected = $signed(OUTPUT_Q_SAMPLES_VEC[OUTPUT_SAMPLES_WIDTH*i +: OUTPUT_SAMPLES_WIDTH])/2.834630687;
          // Check I / Q values, should be a ramp
          $sformat(s, "Incorrect I value received! Expected: %0d, Received: %0d",
            i_expected, i_samp);
          `ASSERT_ERROR((i_samp == i_expected) || (i_samp-1 == i_expected) || (i_samp+1 == i_expected), s);
          $sformat(s, "Incorrect Q value received! Expected: %0d, Received: %0d",
            q_expected, q_samp);
          `ASSERT_ERROR((q_samp == q_expected) || (q_samp-1 == q_expected) || (q_samp+1 == q_expected), s);
          // Check tlast
          if (i == 0) begin
            `ASSERT_ERROR(last, "Last not asserted on final word!");
          end else begin
            `ASSERT_ERROR(~last, "Last asserted early!");
          end
        end
      end
    join
    `TEST_CASE_DONE(1);
    `TEST_BENCH_DONE;

  end
endmodule
