`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: LineBuffer
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

module LineBuffer #(
    parameter IX = 32,
    parameter IY = 32
)(
    input clk,
    input reset_n,

    input i_in_valid,
    // 32 bit 1pixel 씩 받음
    input [`IF_BW-1 : 0] i_in_pixel,

    output o_window_valid,
    output [`POOL_IN_SIZE*`POOL_IN_SIZE*`IF_BW-1 : 0] o_window
    );

    parameter LATENCY = 1;
    reg [$clog2(IX)-1:0] x_cnt;
    reg [$clog2(IY)-1:0] y_cnt;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <=0;
            y_cnt <=0;
        end else if (i_in_valid) begin
            if (x_cnt < IX ) begin
                x_cnt <=0;
                if (y_cnt < IY) begin
                    y_cnt <=0; 
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
               x_cnt <= x_cnt + 1;
            end
        end
    end
endmodule
