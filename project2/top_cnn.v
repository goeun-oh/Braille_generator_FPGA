`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 02:37:58
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

module top_cnn (
    `include "defines_cnn_core.vh"
    input  wire clk,
    input  wire rst_n,
    input  wire i_valid,
    input  wire [`CI*`X_SIZE*`X_SIZE*`IF_BW-1:0] i_conv_out,  // Conv2 결과
    input  wire [`FC_OUT*`FLAT_SIZE*`W_BW-1:0]   i_fc_weight,
    input  wire [`FC_OUT*`ACC_BW-1:0]            i_fc_bias,
    output wire [`FC_OUT*`ACC_BW-1:0]            o_fc_out,
    output wire                                  o_done
);

    /* 1) MaxPool -------------------------------------*/
    wire                      v_pool;
    wire [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0] pool_out;
    maxpool_layer u_pool(
        .clk(clk), .rst_n(rst_n),
        .i_valid(i_valid),
        .i_ifmap(i_conv_out),
        .o_valid(v_pool),
        .o_ofmap(pool_out)
    );

    /* 2) Flatten -------------------------------------*/
    wire                      v_flat;
    wire [`FLAT_SIZE*`OF_BW-1:0] flat_out;
    flatten u_flat(
        .clk(clk), .rst_n(rst_n),
        .i_valid(v_pool),
        .i_map3d(pool_out),
        .o_valid(v_flat),
        .o_vec1d(flat_out)
    );

    /* 3) Fully-Connected -----------------------------*/
    wire [`FC_OUT-1:0] v_fc;
    fc_layer u_fc(
        .clk(clk), .rst_n(rst_n),
        .i_valid(v_flat),
        .i_flat(flat_out),
        .i_all_weight(i_fc_weight),
        .i_all_bias(i_fc_bias),
        .o_all_out(o_fc_out),
        .o_all_valid(v_fc)
    );

    assign o_done = &v_fc;  // 모든 뉴런 valid 시 완료
endmodule
