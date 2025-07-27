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
`include "stage3_defines_cnn_core.vh"
module stage3_top_cnn(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`stage2_CI * `IF_BW - 1: 0] i_in_Relu,

    output o_valid,
    output [7:0] alpha
    );

    wire pool_valid;
    wire [`pool_CO * `OF_BW-1:0] w_pool;
    wire acc_valid;
    wire [`acc_CO * `ACC_BW-1:0] w_acc;
    wire core_valid;
    wire [`core_CO * `OUT_BW -1:0] w_core;

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

    stage3_cnn_core U_stage3_cnn_core(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(acc_valid),
    .o_ot_ci_acc(w_acc),
    .o_ot_valid(core_valid),
    .o_ot_result(w_core)
    );
    
    stage3_compare_alpha U_stage3_compare_alpha(
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(core_valid),
        .i_in_core(w_core),
        .o_alpha(alpha),
        .o_valid(o_valid)
    );

endmodule

// a = 0x61 b = 0x62, c = 0x63

module stage3_compare_alpha (
    input clk,
    input reset_n,
    input i_in_valid,
    input [`core_CO * `OUT_BW -1:0] i_in_core,
    output reg [7:0] o_alpha,
    output o_valid
);
    localparam LATENCY = 1;
    reg signed [`OUT_BW - 1:0] c_ot_result [0 : `core_CO-1];

    reg  signed   [LATENCY - 1 : 0]         r_valid;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= 0;
        end else begin
            r_valid[LATENCY - 1]  <= i_in_valid;
            // r_valid[LATENCY - 2]  <= i_in_valid;
            // r_valid[LATENCY - 1]  <= r_valid[LATENCY - 2];
        end
    end

    integer i;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            for (i=0;i<`core_CO;i=i+1) begin
                c_ot_result[i] <= 0;
            end
        end else if (i_in_valid) begin
            for (i=0;i<`core_CO;i=i+1) begin
                c_ot_result[i] <= $signed(i_in_core [i*`OUT_BW+:`OUT_BW]);
            end
        end
    end

    always @(*) begin
        if ((c_ot_result[0] >= c_ot_result[1]) && (c_ot_result[0] >= c_ot_result[2])) begin
            o_alpha = 8'h61;
        end else if ((c_ot_result[1] >= c_ot_result[0]) && (c_ot_result[1] >= c_ot_result[2])) begin
            o_alpha = 8'h62;
        end else begin
            o_alpha = 8'h63;
        end
    end

    assign o_valid = r_valid[LATENCY - 1];


endmodule