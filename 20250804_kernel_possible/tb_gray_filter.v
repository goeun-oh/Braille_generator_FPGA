`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/25 18:43:38
// Design Name: 
// Module Name: tb_gray_filter
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

`include "gray_filter.v"  // gray_filter 모듈 포함

module tb_gray_filter;
    reg clk;
    reg reset_n;
    reg [`I_DATA_BW-1:0] one_px;
    reg i_in_valid;
    wire [8-1:0] grayed_one_px;
    wire o_valid;

    // DUT 인스턴스
    gray_filter dut (
        .clk(clk),
        .reset_n(reset_n),
        .one_px(one_px),
        .i_in_valid(i_in_valid),
        .grayed_one_px(grayed_one_px),
        .o_valid(o_valid)
    );

    // Clock 생성
    always #5 clk = ~clk;

    // File 관련 변수
    integer i, fp;
    reg [32-1:0] input_data [0:783]; // 28x28 = 784
    initial begin 
        $readmemh("a_1_rgb.mem", input_data);
    end

    initial begin
        clk = 0;
        i_in_valid = 0;
        reset_n = 0;
        #10;
        reset_n = 1;

        fp = $fopen("a_1_gray_verilog.mem", "w");

        for (i = 0; i < 784; i = i + 1) begin
            @(posedge clk);
            one_px <= input_data[i];
            i_in_valid <= 1;
            @(posedge clk);
            $fwrite(fp, "0x%02x\n", grayed_one_px);
         
        end

        @(posedge clk);
        i_in_valid <= 0;

        $fclose(fp);
        $display("✅ Grayscale values saved to gray_output.mem");
        $finish;
    end
endmodule
