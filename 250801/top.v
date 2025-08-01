`timescale 1ns/1ps

`include "defines_cnn_core.v"

module top(
    input clk,
    input i_btn,
    input reset,
    input [3:0] sw,
    output [2:0] led
);


    wire w_btn;
    // wire clk_out;

    // clk_div5 u_clk_div5(
    // .clk(clk),        // 100MHz 입력 클럭
    // .reset_n(!reset),    // 비동기 리셋 (active low)
    // .clk_out(clk_out)  
    // );

    


    
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
        .out_valid(),
        .alpha(),
        .led(led)
    );


endmodule

// module clk_div5 (
//     input  wire clk,        // 100MHz 입력 클럭
//     input  wire reset_n,    // 비동기 리셋 (active low)
//     output reg  clk_out     // 20MHz 출력 클럭
// );

//     reg [2:0] cnt;

//     always @(posedge clk or negedge reset_n) begin
//         if (!reset_n) begin
//             cnt     <= 3'd0;
//             clk_out <= 1'b0;
//         end else begin
//             if (cnt == 3'd4) begin
//                 clk_out <= ~clk_out;
//                 cnt <= 0;
//             end else begin
//                 cnt <= cnt + 1;
//             end
//         end
//     end

// endmodule

