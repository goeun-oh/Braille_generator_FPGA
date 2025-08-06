`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/23 12:35:49
// Design Name: 
// Module Name: top_cnn
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: ST3_W_BW
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include "defines_cnn_core.v"


module stage3_top_cnn(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`stage3_CI * `ST3_IF_BW - 1: 0] i_in_Relu,

    output o_valid,
    output [7:0] alpha,
    output [3:0] led,
    output led_r,
    output led_g,
    output led_b
    );
    wire pool_valid;
    wire [`pool_CO * `ST3_OF_BW-1:0] w_pool;
    wire acc_valid;
    wire [`acc_CO * `ST3_ACC_BW-1:0] w_acc;
    wire core_valid;
    wire [`core_CO * `ST3_OUT_BW -1:0] w_core;


    // 확인 완료
    stage3_max_pooling U_stage3_max_pooling(
    .clk(clk),
    .reset_n(reset_n),
    .i_Relu_valid(i_Relu_valid),
    .i_in_Relu(i_in_Relu),
    .o_ot_valid(pool_valid),
    .o_ot_pool(w_pool)
    );

    stage3_cnn_acc_ci U_stage3_cnn_acc_ci(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(pool_valid),
    .i_in_pooling(w_pool),
    .o_ot_valid(acc_valid),
    .o_ot_ci_acc(w_acc)
    );

    wire [`ST3_OUT_BW-1:0] w_cnn_value;
    wire [$clog2(`acc_CO)-1:0] w_index_info;   
    wire [$clog2(`acc_CO)-1:0] max_index_info;   
    wire w_compare_valid; 

    stage3_cnn_core U_stage3_cnn_core(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(acc_valid),
    .o_ot_ci_acc(w_acc),
    .o_ot_valid(core_valid),
    .o_cnn_value(w_cnn_value),
    .index_info(w_index_info)
    );

    compare_alpha U_compare_alpha(
        .clk(clk),
        .reset_n(reset_n),
        .i_cnn_value(w_cnn_value),
        .i_index_info(w_index_info),
        .i_valid(core_valid),
        .o_valid(w_compare_valid),
        .o_index_info(max_index_info)
    );

    alpha_decoder u_alpha_decoder(
        .clk(clk),
        .reset_n(reset_n),
        .i_valid(w_compare_valid),
        .index_info(max_index_info),
        .o_alpha(alpha),
        .o_valid(o_valid),
        .led(led),
        .led_r(led_r),
        .led_g(led_g),
        .led_b(led_b)
    );


endmodule

