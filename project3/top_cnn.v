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

module top_cnn(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`CI * `IF_BW - 1: 0] i_in_Relu,

    output o_ot_valid,
    output [`CO * `OUT_BW -1:0] o_ot_result
    );

    wire pool_valid;
    wire [`CO * `OF_BW-1:0] w_pool;
    wire acc_valid;
    wire [`CO * `ACC_BW-1:0] w_acc;


    max_pooling U_max_pooling(
    .clk(clk),
    .reset_n(reset_n),
    .i_Relu_valid(i_Relu_valid),
    .i_in_Relu(i_in_Relu),
    .o_ot_valid(pool_valid),
    .o_ot_pool(w_pool)
    );

    cnn_acc_ci U_cnn_acc_ci(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(pool_valid),
    .i_in_pooling(w_pool),
    .o_ot_valid(acc_valid),
    .o_ot_ci_acc(w_acc)
    );

    cnn_core U_cnn_core(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(acc_valid),
    .o_ot_ci_acc(w_acc),
    .o_ot_valid(o_ot_valid),
    .o_ot_result(o_ot_result)
    );
endmodule
