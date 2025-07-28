`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: cnn_kernal
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

module stage3_cnn_kernal(
    input wire clk,
    input wire reset_n,

    input wire i_pooling_valid,
    input wire [`pool_CI * `OF_BW-1:0] i_pooling,
    input wire [`pool_CI * `W_BW - 1 : 0] i_weight,

    output wire o_kernal_valid,
    output wire [`MUL_BW + $clog2(3) - 1: 0]o_kernel
    // 값 확인용

    );

    localparam LATENCY = 4; // Increased latency due to additional input pipeline stage

    wire   [LATENCY-1 : 0]    ce;
    reg    [LATENCY-1 : 0]    r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= {LATENCY{1'b0}};
        end else begin
            r_valid[0]  <= i_pooling_valid;
            r_valid[1]  <= r_valid[0];
            r_valid[2]  <= r_valid[1];
            r_valid[3]  <= r_valid[2];
        end
    end
    assign   ce = r_valid;

    // Stage 1: Input capture and unpacking
    reg [`OF_BW-1:0] pool_ch[0:`pool_CI-1];
    reg [`W_BW-1:0]  weight_ch[0:`pool_CI-1];

    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < `pool_CI; i = i + 1) begin
                pool_ch[i]   <= 0;
                weight_ch[i] <= 0;
            end
        end else if (i_pooling_valid) begin
            for (i = 0; i < `pool_CI; i = i + 1) begin
                pool_ch[i]   <= i_pooling[i * `OF_BW +: `OF_BW];
                weight_ch[i] <= i_weight[i * `W_BW +: `W_BW];
            end
        end
    end

    // Stage 2: DSP Input Pipeline Stage (A and B inputs)
    reg signed [`OF_BW-1:0] r_pool_ch[`pool_CI-1:0];
    reg signed [`W_BW-1:0]  r_weight_ch[`pool_CI-1:0];

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < `pool_CI; i = i + 1) begin
                r_pool_ch[i]   <= 0;
                r_weight_ch[i] <= 0;
            end
        end else if (r_valid[0]) begin
            for (i = 0; i < `pool_CI; i = i + 1) begin
                r_pool_ch[i]   <= pool_ch[i];
                r_weight_ch[i] <= weight_ch[i];
            end
        end
    end

    // Stage 3: DSP Multiplication with properly pipelined inputs
    reg signed [`MUL_BW-1:0] r_mul [0:`pool_CI-1];
    genvar mul_idx;
    generate
        for (mul_idx = 0; mul_idx < `pool_CI; mul_idx = mul_idx + 1) begin : gen_mul
            (* use_dsp = "yes" *)
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n)
                    r_mul[mul_idx] <= 0;
                else if (r_valid[1])  // Using the pipelined inputs
                    r_mul[mul_idx] <= $signed(r_pool_ch[mul_idx]) * $signed(r_weight_ch[mul_idx]);
            end
        end
    endgenerate

    // Stage 4: Accumulation
    reg signed [`MUL_BW + $clog2(3) - 1: 0] r_acc_kernel;
    reg signed [`MUL_BW + $clog2(3) - 1:0] acc_temp;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_acc_kernel <= 0;
        end else if (r_valid[2]) begin
            acc_temp = 0;
            for (i = 0; i < `pool_CI; i = i + 1) begin
                acc_temp = acc_temp + r_mul[i];
            end
            r_acc_kernel <= acc_temp;
        end
    end

    assign o_kernal_valid = r_valid[LATENCY-1];
    assign o_kernel = r_acc_kernel;

endmodule