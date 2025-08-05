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
`include "defines_cnn_core.v"

module stage3_cnn_kernal(
    input wire clk,
    input wire reset_n,

    input wire i_pooling_valid,
    input wire [`pool_CO * `ST3_OF_BW-1:0] i_pooling,
    input wire [`pool_CO * `ST3_W_BW - 1 : 0] i_weight,

    output wire o_kernal_valid,
    output wire [`ST3_MUL_BW + $clog2(`pool_CO) - 1: 0]o_kernel
    // 값 확인용

    );

    localparam LATENCY = 3;

    wire   [LATENCY-1 : 0] 	ce;
    reg    [LATENCY-1 : 0] 	r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-3]  <= i_pooling_valid;
            r_valid[LATENCY-2]  <= r_valid[LATENCY-3];
            r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
        end
    end
    assign	ce = r_valid;


    // reducing fanout by pipelining
    reg [`ST3_OF_BW-1:0] pool_ch[0:`pool_CO-1];
    reg [`ST3_W_BW-1:0]  weight_ch[0:`pool_CO-1];

    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < `pool_CO; i = i + 1) begin
                pool_ch[i]   <= 0;
                weight_ch[i] <= 0;
            end
        end else if (i_pooling_valid) begin
            for (i = 0; i < `pool_CO; i = i + 1) begin
                pool_ch[i]   <= i_pooling[i * `ST3_OF_BW +: `ST3_OF_BW];
                weight_ch[i] <= i_weight[i * `ST3_W_BW +: `ST3_W_BW];
            end
        end
    end


    wire  signed    [`pool_CO * `ST3_MUL_BW-1 : 0]    mul  ;
    reg   signed    [`pool_CO * `ST3_MUL_BW-1 : 0]    r_mul;
    
   reg [`ST3_MUL_BW-1 : 0] d_mul;

    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            d_mul <= 0;
        end else begin
            d_mul <= r_mul[0 +: `ST3_MUL_BW];
        end
    end

    genvar mul_idx;
    generate
        for(mul_idx = 0; mul_idx < `pool_CO; mul_idx = mul_idx + 1) begin : gen_mul
            assign  mul[mul_idx * `ST3_MUL_BW +: `ST3_MUL_BW]	=  $signed(pool_ch[mul_idx]) * $signed(weight_ch[mul_idx]);
        
            always @(posedge clk or negedge reset_n) begin
                if(!reset_n) begin
                    r_mul[mul_idx * `ST3_MUL_BW +: `ST3_MUL_BW] <= 0;
                end else if(r_valid[LATENCY-3])begin
                    r_mul[mul_idx * `ST3_MUL_BW +: `ST3_MUL_BW] <= $signed(mul[mul_idx * `ST3_MUL_BW +: `ST3_MUL_BW]);
                end
            end
        end
    endgenerate 
    
    reg signed [`ST3_MUL_BW + $clog2(`pool_CO) - 1: 0] acc_kernel 	;
    reg signed [`ST3_MUL_BW + $clog2(`pool_CO) - 1: 0] r_acc_kernel   ;

    integer acc_idx;
    always @ (*) begin
        acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))]= 0;
        for(acc_idx =0; acc_idx < `pool_CO; acc_idx = acc_idx +1) begin
            acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))] = $signed(acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))]) + $signed(r_mul[acc_idx*`ST3_MUL_BW +: `ST3_MUL_BW]); 
        end
    end
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))] <= 0;
        end else if(r_valid[LATENCY-2])begin
            r_acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))] <= $signed(acc_kernel[0 +: (`ST3_MUL_BW + $clog2(`pool_CO))]);
        end
    end



    assign o_kernal_valid = r_valid[LATENCY-1];
    assign o_kernel = r_acc_kernel;

endmodule