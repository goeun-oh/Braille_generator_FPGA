`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 10:25:11
// Design Name: 
// Module Name: conv2_bias_rom
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

module conv2_bias_rom(
    output reg signed [`ST2_Conv_CO*`B_BW -1  : 0] bias   
    );

    localparam TOTAL_BIAS = `ST2_Conv_CO; 

    reg signed [`B_BW-1:0] bias_mem [0:TOTAL_BIAS-1];              

    initial begin
        $readmemh("conv2_bias.mem", bias_mem);
    end

    integer i;
    always @(*) begin
        for (i = 0; i < TOTAL_BIAS; i = i + 1) begin
            bias[i*`B_BW +: `B_BW] = bias_mem[i];
        end
    end

endmodule