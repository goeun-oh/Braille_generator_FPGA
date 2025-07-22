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
`include "defines_cnn_core.vh"

module max_pooling(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`CI * `IF_BW - 1: 0] i_in_Relu,

    output wire o_ot_valid,
    output wire [`FC_IN_VEC * `OF_BW - 1 : 0] o_ot_pool
    );

    wire [`CI-1 : 0] w_ot_valid;
    // 3 * 2 * 2 * 32
    wire [`CI * `POOL_K*`POOL_K*`IF_BW-1:0] w_ot_window;
    // 48 * 32
    wire [`CO * `OF_BW-1:0] w_ot_pool;
    reg [`FC_IN_VEC * `IF_BW -1 : 0]  w_ot_flat;
    reg [$clog2(`P_SIZE * `P_SIZE)-1:0] flat_cnt;
    genvar line_inst;
    generate
        for (line_inst = 0; line_inst < `CI ; line_inst = line_inst + 1) begin
            wire [`IF_BW - 1: 0] w_in_pixel = i_in_Relu[line_inst * `IF_BW +: `IF_BW];
            line_buffer U_line_buffer(
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
        for (pool_inst = 0; pool_inst < `CI ; pool_inst = pool_inst + 1) begin
            max_pool_2x2 U_max_pool (
                .i00(w_ot_window[pool_inst * `POOL_K * `POOL_K * `IF_BW +: `IF_BW]),
                .i01(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 1) * `IF_BW +: `IF_BW]),
                .i10(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 2) * `IF_BW +: `IF_BW]),
                .i11(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 3) * `IF_BW +: `IF_BW]),
                .o_max(w_ot_pool[pool_inst * `OF_BW +: `OF_BW])
            );
        end
    endgenerate


    integer i;
    always @(posedge clk, posedge reset_n) begin
        if (!reset_n) begin
            w_ot_flat <= 0;
        end else if (&w_ot_valid) begin
            for(i=0 ; i < `CO; i = i + 1) begin
                w_ot_flat[((i*`P_SIZE*`P_SIZE)+flat_cnt)*`OF_BW +: `OF_BW] <= w_ot_pool[i * `OF_BW +: `OF_BW];
            end
            flat_cnt = flat_cnt + 1;
        end
    end

    

endmodule

module max_pool_2x2 (
    input  [`OF_BW-1:0] i00, i01, i10, i11,
    output [`OF_BW-1:0] o_max
);
    wire [`OF_BW-1:0] max0 = ($signed(i00) > $signed(i01)) ? i00 : i01;
    wire [`OF_BW-1:0] max1 = ($signed(i10) > $signed(i11)) ? i10 : i11;
    assign o_max = ($signed(max0) > $signed(max1)) ? max0 : max1;
endmodule