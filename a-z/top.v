`timescale 1ns/1ps

`include "defines_cnn_core.v"

module top(
    input clk,
    input i_btn,
    input reset,
    input [3:0] sw,
    output [7:0] alpha,
    output out_valid
);


    wire w_btn;

    btn_debounce_one_pulse U_BTN(
        .clk(clk),
        .reset_n(!reset),
        .i_btn(i_btn),
        .o_btn(w_btn)
    );

    cnn_top U_cnn_top(
        .clk(clk),
        .reset_n(!reset),
        .i_valid(w_btn),
        .sw(sw),
        .out_valid(out_valid),
        .alpha(alpha)
    );


endmodule
