`timescale 1ns/1ps

`include "defines_cnn_core.v"

module top(
    input clk,
    input valid_cnn,
    input [31:0] pcam_data,
    input reset,
    output [2:0] led,
    output [7:0] alpha,
    output out_valid
);

    cnn_top U_cnn_top(
        .clk(clk),
        .reset_n(reset),
        .i_valid(valid_cnn),
        .i_pixel(pcam_data),
        .out_valid(out_valid),
        .alpha(alpha),
        .led(led)
    );


endmodule
