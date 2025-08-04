`timescale 1ns / 1ps

`include "defines_cnn_core.v"
module conv2_bias_rom(
    output reg signed [`ST2_Conv_CO*`ST2_B_BW -1  : 0] bias   
    );

    localparam TOTAL_BIAS = `ST2_Conv_CO; 

    reg signed [`ST2_B_BW-1:0] bias_mem [0:TOTAL_BIAS-1];              

    initial begin
        $readmemh("conv2_bias.mem", bias_mem);
    end

    integer i;
    always @(*) begin
        for (i = 0; i < TOTAL_BIAS; i = i + 1) begin
            bias[i*`ST2_B_BW +: `ST2_B_BW] = bias_mem[i];
        end
    end

endmodule