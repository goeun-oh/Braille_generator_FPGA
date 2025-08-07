`timescale 1ns/1ps

`include "defines_cnn_core.v"

module top(
    input clk,
    input reset_n,
    input valid_cnn,
    input [31:0] pcam_data,
    input [3:0] sw,
    output [7:0] alpha,
    output out_valid,
    output [3:0] led,
    output led_r,
    output led_g,
    output led_b
);


    wire w_valid;

    cnn_top U_cnn_top(
        .clk(clk),
        .reset_n(!reset),
        .i_valid(valid_cnn),
        .i_pixel(pcam_data),
        .sw(sw),
        .out_valid(w_valid),
        .alpha(alpha),
        .led(led),
        .led_r(led_r),
        .led_g(led_g),
        .led_b(led_b)
    );

    valid_gen u_valid_gen(
        .clk(clk),
        .reset_n(!reset),
        .i_valid(w_valid),
        .o_valid(out_valid)
    );

endmodule

