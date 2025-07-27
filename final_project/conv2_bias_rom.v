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

module conv2_bias_rom#(
    parameter CHANNEL_ID = 0
)(
    output reg signed[`B_BW-1:0] bias_out
);

always @(*) begin
    case (CHANNEL_ID)
        0: bias_out = 6'sd2;
        1: bias_out = -6'sd23;
        2: bias_out = 6'sd1;
        default: bias_out = 30'd0;
    endcase
end

endmodule