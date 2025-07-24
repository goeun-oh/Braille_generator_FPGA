`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 07:17:32
// Design Name: 
// Module Name: conv2_weight_rom
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


module conv2_weight_rom(
    output reg signed [`ST2_Conv_CO*`ST2_Conv_CI * `KX * `KY * `W_BW-1:0] weight   //3x3x5x5x(7bit) 중 1개를 3x5x5x(7bit)로 출력
    );

    localparam TOTAL_WEIGHT = `ST2_Conv_CO * `ST2_Conv_CI * `KX * `KY; // 3*3*5*5 = 225


    // 8bit 3x3×5×5 = 225 weight (8bit signed each)
    reg signed [`W_BW-1:0] weight_mem [0:TOTAL_WEIGHT-1];              //8bit 225개


    initial begin
        $readmemh("conv2_weights.mem", weight_mem);
    end

    integer i;
    always @(*) begin
        for (i = 0; i < TOTAL_WEIGHT; i = i + 1) begin
            weight[i*`W_BW +: `W_BW] = weight_mem[i];
        end
    end


endmodule
