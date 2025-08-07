`timescale 1ns / 1ps
`include "defines_cnn_core.v"
module cnn_top_tb;
    parameter CLK_PERIOD = 10;

    // Clock & Reset
    reg clk = 0;
    reg reset_n = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // DUT inputs
    reg                      i_valid;
    reg  [`ST1_I_F_BW-1:0]        i_pixel;
    reg [3:0] sw_val;


    wire core_done;
    wire [7:0] alpha;

    cnn_top dut (
        .clk(clk),
        .reset_n(reset_n),
        .i_valid(i_valid),
        .sw(sw_val),
        .out_valid(core_done),
        .alpha(alpha)
    );

    // === 테스트 시나리오 ===
    integer i;
    integer row, col, idx;
    reg [$clog2(`ST1_IX)-1:0]cnt;
    initial begin
        // 초기화
        reset_n = 0;
        i_valid = 0;
        i_pixel = 0;
        cnt =0;
        #100;
        reset_n = 1;
        #10

        // sw_val = 4;
        for (sw_val = 0; sw_val < 12; sw_val = sw_val + 1) begin
            $display("\n=== [TEST] sw = %0d ===", sw_val);
            
            @(posedge clk);
            i_valid = 1;
            #100;
            @(posedge clk);
            i_valid = 0;

            // 결과 나올 때까지 기다리기
            wait(core_done==1)

            // 결과 출력 (또는 저장)
            #10000;
        end

        $finish;
    end

    always @(posedge clk) begin
        if (core_done) begin

            $display("predicted alphabet is : %c", alpha);
        end
    end

endmodule