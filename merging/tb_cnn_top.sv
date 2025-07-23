`timescale 1ns / 1ps

module cnn_top_tb;
    parameter CLK_PERIOD = 10;
    parameter I_F_BW = 8;
    parameter O_F_BW = 20;
    parameter KX = 5;
    parameter KY = 5;
    parameter W_BW = 8;
    parameter CI = 1;
    parameter CO = 3;
    parameter IX = 28;
    parameter IY = 28;
    parameter OUT_W = IX - KX + 1;
    parameter OUT_H = IY - KY + 1;

    // Clock & Reset
    reg clk = 0;
    reg reset_n = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // DUT inputs
    reg                      i_valid;
    reg  [I_F_BW-1:0]        i_pixel;

    // DUT output
    wire pooling_core_valid;
    wire [3 * 19-1:0] pooling_core_fmap;

    //wire [KX*KY*I_F_BW-1:0] o_window;
    //wire [KX*I_F_BW-1:0] o_line_buf;
    cnn_top #(
        .I_F_BW(I_F_BW), .O_F_BW(O_F_BW), .KX(KX), .KY(KY),
        .CI(CI), .CO(CO), .IX(IX), .IY(IY)
    ) dut (
        .clk(clk),
        .reset_n(reset_n),
        .i_valid(i_valid),
        .w_pooling_core_valid(pooling_core_valid),
        .w_pooling_core_fmap(pooling_core_fmap)
    );

    // === 테스트 시나리오 ===
    integer i;
    integer row, col, idx;
    reg [$clog2(IX)-1:0]cnt;
    initial begin
        // 초기화
        reset_n = 0;
        i_valid = 0;
        i_pixel = 0;
        cnt =0;
        #100;
        reset_n = 1;
        #10

        @(posedge clk);
        i_valid = 1;
        #10;
        @(posedge clk);
        i_valid = 0;
        // === 결과 기다리기 ===

        //for (ch = 0; ch < 3; ch = ch + 1) begin
        //end
    end


    //always @(*) begin
    //    $display("==== o_window 5x5 ====, cnt = %d", cnt);
    //    for (row = 0; row < KX; row = row + 1) begin
    //        $display("--- row %0d ---", row);
    //        for (col = 0; col < KY; col = col + 1) begin
    //            idx = row*KX + col;
    //            $write("(%6d) ", o_window[idx*I_F_BW +: I_F_BW]);
    //        end
    //        $write("\n");
    //    end
    //    cnt = cnt + 1;
    //    $finish;
    //end
endmodule
