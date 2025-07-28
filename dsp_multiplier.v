
`timescale 1ns/1ps
module dsp_multiplier #(
    parameter A_BW = 9,    // Input A bit width (unsigned feature map + 1 sign bit)
    parameter B_BW = 8,    // Input B bit width (signed weight)
    parameter P_BW = 16    // Product bit width
)(
    input wire clk,
    input wire reset_n,
    input wire enable,
    input wire signed [A_BW-1:0] a,    // Zero-extended unsigned input
    input wire signed [B_BW-1:0] b,    // Signed weight
    output reg signed [P_BW-1:0] product
);

    (* use_dsp = "yes" *)
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            product <= 0;
        end else if (enable) begin
            product <= a * b;  // Both are now properly signed
        end
    end

endmodule