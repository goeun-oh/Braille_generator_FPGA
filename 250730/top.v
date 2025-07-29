`timescale 1ns/1ps


module top(
    input clk,
    input i_btn,
    input reset,
    input [3:0] sw,
    output [2:0] led
);
    parameter I_F_BW       = 8;
    parameter O_F_BW       = 20;
    parameter KX           = 5;
    parameter KY           = 5;
    parameter W_BW         = 8;
    parameter CI           = 1;
    parameter CO           = 3;
    parameter IX           = 28;
    parameter IY           = 28;
    parameter B_BW         = 16;
    parameter AK_BW        = 21;
    parameter ACI_BW       = 21;
    parameter AB_BW        = 21;
    parameter AR_BW        = 20;


    parameter BITSHIFT_BW  = 12;


    parameter OUT_W        = IX - KX + 1;
    parameter OUT_H        = IY - KY + 1;
    //pooling//
    parameter ST2_Pool_CI  = 3;
    parameter ST2_Pool_CO  = 3;
    parameter ST2_Conv_CI  = 3;
    parameter ST2_Conv_CO  = 3;
    parameter ST2_Conv_IBW = 20;
    parameter ST2_O_F_BW   = 35;
    parameter POOL_OUT_W = 12;
    parameter POOL_OUT_H = 1;

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

    cnn_top #(
        .I_F_BW       (I_F_BW),
        .O_F_BW       (O_F_BW),
        .KX           (KX),
        .KY           (KY),
        .W_BW         (W_BW),
        .CI           (CI),
        .CO           (CO),
        .IX     (IX),
        .IY     (IY),
        .B_BW   (B_BW),
        .AK_BW  (AK_BW),
        .ACI_BW (ACI_BW),
        .AB_BW  (AB_BW),  
        .AR_BW  (AR_BW),
        .OUT_W(OUT_W),
        .OUT_H(OUT_H),
        .ST2_Pool_CI(ST2_Pool_CI),
        .ST2_Pool_CO(ST2_Pool_CO),
        .ST2_Conv_CI(ST2_Conv_CI),
        .ST2_Conv_CO(ST2_Conv_CO),
        .ST2_Conv_IBW (ST2_Conv_IBW),
        .ST2_O_F_BW   (ST2_O_F_BW),
        .POOL_OUT_W (POOL_OUT_W),
        .POOL_OUT_H(POOL_OUT_H)
    ) U_cnn_top(
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

