`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: max_pooling
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include "stage3_defines_cnn_core.vh"

module stage3_max_pooling(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`stage2_CI * `IF_BW - 1: 0] i_in_Relu,

    output wire o_ot_valid,
    output wire [`pool_CO * `OF_BW-1:0] o_ot_pool
    );
    localparam LATENCY = 2;

    wire [`linebuf_CO-1 : 0] w_ot_valid;
    // 3 * 2 * 2 * 32
    wire [`linebuf_CO * `POOL_K*`POOL_K*`IF_BW-1:0] w_ot_window;
    // 48 * 32
    wire [`pool_CO * `OF_BW-1:0] w_ot_pool;
    reg  [`pool_CO * `OF_BW-1:0] w_ot_flat;
    reg r_pooling_valid;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-2]  <= &w_ot_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign	ce = r_valid;


    genvar line_inst;
    generate
        for (line_inst = 0; line_inst < `stage2_CI ; line_inst = line_inst + 1) begin
            wire [`IF_BW - 1: 0] w_in_pixel = i_in_Relu[line_inst * `IF_BW +: `IF_BW];
            stage3_line_buffer U_line_buffer(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(i_Relu_valid),
                .i_in_pixel(w_in_pixel),
                .o_window_valid(w_ot_valid[line_inst]),
                .o_window(w_ot_window[line_inst * `POOL_K * `POOL_K * `IF_BW +: `POOL_K * `POOL_K * `IF_BW])
            );
        end
    endgenerate

    genvar pool_inst;
    generate
        for (pool_inst = 0; pool_inst < `pool_CI ; pool_inst = pool_inst + 1) begin
            stage3_max_pool_2x2 U_max_pool (
                .i00(w_ot_window[pool_inst * `POOL_K * `POOL_K * `IF_BW +: `IF_BW]),
                .i01(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 1) * `IF_BW +: `IF_BW]),
                .i10(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 2) * `IF_BW +: `IF_BW]),
                .i11(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 3) * `IF_BW +: `IF_BW]),
                .o_max(w_ot_pool[pool_inst * `OF_BW +: `OF_BW])
            );
        end
    endgenerate


reg [`pool_CO * `OF_BW-1:0] r_pool_result;
// 1클럭 pipelining
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        r_pool_result <= 0;
    end else if (&w_ot_valid) begin
        r_pool_result <= w_ot_pool;
    end 
end



    // 디버깅용
    reg [`OF_BW -1 : 0]r_o_ot_flat [0:2];
    integer i;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            w_ot_flat <= 0;
            r_pooling_valid <= 0;
        end else if (r_valid[LATENCY-2]) begin
            w_ot_flat <= r_pool_result;
            // // 디버깅용 시작
            // for(i = 0; i< `pool_CO ; i = i + 1) begin
            //     r_o_ot_flat[i] <= w_ot_pool[i * `OF_BW +: `OF_BW];    
            // end
            // // 디버깅용 끝
            // r_pooling_valid <= 1;
        end 
        // else begin
        //     r_pooling_valid <= 0;
        // end
    end

    assign o_ot_pool = w_ot_flat;
    assign o_ot_valid = r_valid[LATENCY-1];

endmodule

module stage3_max_pool_2x2 (
    input  [`OF_BW-1:0] i00, i01, i10, i11,
    output [`OF_BW-1:0] o_max
);
    wire [`OF_BW-1:0] max0 = ($signed(i00) > $signed(i01)) ? i00 : i01;
    wire [`OF_BW-1:0] max1 = ($signed(i10) > $signed(i11)) ? i10 : i11;
    assign o_max = ($signed(max0) > $signed(max1)) ? max0 : max1;
endmodule