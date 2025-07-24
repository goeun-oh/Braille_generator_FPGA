`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: cnn_core
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

module cnn_core(
    input clk,
    input reset_n,

    // pooling valid
    input i_in_valid,
    // 48 개 중 0, 16, 32
    input [`CO * `ACC_BW-1:0] o_ot_ci_acc,
    
    output o_ot_valid,
    output [`CO * `OUT_BW -1:0] o_ot_result
    );
    // bias
    localparam LATENCY = 2;
    reg signed [7:0] bias_mem[0:2];
    reg signed [`CO * `OUT_BW -1:0] w_ot_result;
    reg signed [`CO * `OUT_BW -1:0] r_ot_result;

    reg     [LATENCY - 1 : 0]         r_valid;
    wire    [`CO-1 : 0]         w_ot_valid;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= 0;
        end else begin
            r_valid[LATENCY - 2]  <= i_in_valid;
            r_valid[LATENCY - 1]  <= r_valid[LATENCY - 2];
        end
    end

    initial begin
       $readmemh("fc1_bias.mem", bias_mem);
    end

    integer i;
    always @(*) begin
        w_ot_result = 0;
        for(i=0; i< `CO;i = i + 1) begin
            w_ot_result[i*`OUT_BW+:`OUT_BW] = $signed(o_ot_ci_acc[i*`ACC_BW+:`ACC_BW]) + $signed(bias_mem[i]);
        end 
    end

    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            r_ot_result <= 0;
        end else begin
            r_ot_result <= w_ot_result;
        end
    end
    assign o_ot_valid = r_valid[LATENCY - 1];
    assign o_ot_result = r_ot_result;

endmodule
