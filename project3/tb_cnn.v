`timescale 1ns/1ps

`include "defines_cnn_core.vh"

module tb_top_cnn();

reg clk;
reg reset_n;
reg i_Relu_valid;
reg [`CI * `IF_BW - 1:0] i_in_Relu;

wire o_ot_valid;
wire [`CO * `OUT_BW -1:0] o_ot_result;

top_cnn DUT(
    .clk(clk),
    .reset_n(reset_n),
    .i_Relu_valid(i_Relu_valid),
    .i_in_Relu(i_in_Relu),
    .o_ot_valid(o_ot_valid),
    .o_ot_result(o_ot_result)
);

initial clk = 0;
always #5 clk = ~clk; // 100MHz

reg [`IF_BW-1:0] data_ch0, data_ch1, data_ch2;
integer x, y;

initial begin
    reset_n = 0;
    i_Relu_valid = 0;
    i_in_Relu = 0;
    #30; reset_n = 1; #10;

    // 3채널 8x8(64프레임) 입력
for (y=0; y<`POOL_IN_SIZE; y=y+1) begin
    for (x=0; x<`POOL_IN_SIZE; x=x+1) begin
        @(negedge clk);
        i_Relu_valid = 1;
        // 모든 채널 값 1로 통일!
        data_ch0 = 1;
        data_ch1 = 1;
        data_ch2 = 1;
        i_in_Relu = {data_ch2, data_ch1, data_ch0}; 
    end
end

    @(negedge clk);
    i_Relu_valid = 0;
    i_in_Relu = 0;

    // 출력 대기
    wait(o_ot_valid);
    $display("o_ot_result = %d", o_ot_result);

    repeat(5) @(posedge clk);
    $finish;
end

// 출력 발생시마다 결과 디스플레이
always @(posedge clk) begin
    if (o_ot_valid) begin
        $display("At time %t, o_ot_result = %d", $time, o_ot_result);
    end
end

endmodule
