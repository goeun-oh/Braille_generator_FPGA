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
`include "stage3_efines_cnn_core.vh"

module cnn_kernal(
    input wire clk,
    input wire reset_n,

    input wire i_pooling_valid,
    input wire [`CI * `OF_BW-1:0] i_pooling,
    input wire [`CI * `W_BW - 1 : 0] i_weight,

    output wire o_kernal_valid,
    output wire [`MUL_BW + $clog2(3) - 1: 0]o_kernel
    // 값 확인용

    );

    localparam LATENCY = 2;

    wire   [LATENCY-1 : 0] 	ce;
    reg    [LATENCY-1 : 0] 	r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-2]  <= i_pooling_valid;
            r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
        end
    end
    assign	ce = r_valid;

    wire  signed    [`CI * `MUL_BW-1 : 0]    mul  ;
    reg   signed    [`CI * `MUL_BW-1 : 0]    r_mul;

    genvar mul_idx;
    generate
        for(mul_idx = 0; mul_idx < `CI; mul_idx = mul_idx + 1) begin : gen_mul
            assign  mul[mul_idx * `MUL_BW +: `MUL_BW]	=  $signed(i_pooling[mul_idx * `OF_BW +: `OF_BW]) * $signed(i_weight[mul_idx * `W_BW +: `W_BW ]);
        
            always @(posedge clk or negedge reset_n) begin
                if(!reset_n) begin
                    r_mul[mul_idx * `MUL_BW +: `MUL_BW] <= 0;
                end else if(i_pooling_valid)begin
                    r_mul[mul_idx * `MUL_BW +: `MUL_BW] <= $signed(mul[mul_idx * `MUL_BW +: `MUL_BW]);
                end
            end
        end
    endgenerate 
    
    reg signed [`MUL_BW + $clog2(3) - 1: 0] acc_kernel 	;
    reg signed [`MUL_BW + $clog2(3) - 1: 0] r_acc_kernel   ;

    integer acc_idx;
    generate
        always @ (*) begin
            acc_kernel[0 +: (`MUL_BW + $clog2(3))]= 0;
            for(acc_idx =0; acc_idx < `CI; acc_idx = acc_idx +1) begin
                acc_kernel[0 +: (`MUL_BW + $clog2(3))] = $signed(acc_kernel[0 +: (`MUL_BW + $clog2(3))]) + $signed(r_mul[acc_idx*`MUL_BW +: `MUL_BW]); 
            end
        end
        always @(posedge clk or negedge reset_n) begin
            if(!reset_n) begin
                r_acc_kernel[0 +: (`MUL_BW + $clog2(3))] <= 0;
            end else if(ce[LATENCY-2])begin
                r_acc_kernel[0 +: (`MUL_BW + $clog2(3))] <= $signed(acc_kernel[0 +: (`MUL_BW + $clog2(3))]);
            end
        end
    endgenerate



    assign o_kernal_valid = r_valid[LATENCY-1];
    assign o_kernel = r_acc_kernel;

endmodule
