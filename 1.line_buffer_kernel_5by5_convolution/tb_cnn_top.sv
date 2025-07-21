`timescale 1ns / 1ps

module cnn_top_tb;

    parameter CLK_PERIOD = 10;
    parameter I_F_BW = 8;
    parameter O_F_BW = 23;
    parameter KX = 5;
    parameter KY = 5;
    parameter W_BW = 7;
    parameter CI = 1;
    parameter CO = 1;
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
    reg  [CO*CI*KX*KY*7-1:0] i_cnn_weight;
    reg  [CO*7-1:0]          i_cnn_bias;

    // DUT output
    wire                     o_done;

    wire [KX*KY*I_F_BW-1:0] o_window;
    wire [KX*I_F_BW-1:0] o_line_buf;
    cnn_top #(
        .I_F_BW(I_F_BW), .O_F_BW(O_F_BW), .KX(KX), .KY(KY),
        .CI(CI), .CO(CO), .IX(IX), .IY(IY)
    ) dut (
        .clk(clk),
        .reset_n(reset_n),
        .i_valid(i_valid),
        .i_pixel(i_pixel),
        .i_cnn_weight(i_cnn_weight),
        .i_cnn_bias(i_cnn_bias),
        .o_done(o_done),
        .o_window(o_window),
        .o_line_buf(o_line_buf)
    );

    // === 테스트 시나리오 ===
    integer i;
    integer k;
    integer row, col, idx;
    reg [7:0] image_mem [0:783]; // 28x28
    reg [$clog2(IX)-1:0]cnt;
    initial begin
        // 초기화
        reset_n = 0;
        i_valid = 0;
        i_pixel = 0;
        i_cnn_weight = 0;
        i_cnn_bias = 0;
        cnt =0;
        #100;
        reset_n = 1;
        // 입력 이미지 초기화 (예: 1~784)
        for (i = 0; i < 784; i = i + 1) begin
            image_mem[i] = i + 1; // 1~784
        end

        // Weight 초기화: 모두 1
        for (k = 0; k < 25; k = k + 1) begin
            i_cnn_weight[k*3*7 +: 7]     = 7'd1; // 채널 0
            i_cnn_weight[k*3*7 + 7 +: 7] = 7'd1; // 채널 1
            i_cnn_weight[k*3*7 +14 +: 7] = 7'd1; // 채널 2
        end
        // Bias 초기화: 모두 0
        i_cnn_bias = 0;

        // === 이미지 입력 ===
        @(posedge clk);
        for (i = 0; i < 784; i = i + 1) begin
            i_valid <= 1;
            i_pixel <= image_mem[i];
            @(posedge clk);
        end
        @(posedge clk);
        i_valid <= 0;
        // === 결과 기다리기 ===

        
        wait (o_done);
        //for (ch = 0; ch < 3; ch = ch + 1) begin
        //end
        $display("✅ All convolution outputs done.");
        $finish;
    end


    always @(*) begin
        $display("==== o_window 5x5 ====, cnt = %d", cnt);
        for (row = 0; row < KX; row = row + 1) begin
            $display("--- row %0d ---", row);
            for (col = 0; col < KY; col = col + 1) begin
                idx = row*KX + col;
                $write("(%6d) ", o_window[idx*I_F_BW +: I_F_BW]);
            end
            $write("\n");
        end
        cnt = cnt + 1;
        $finish;
    end
endmodule
