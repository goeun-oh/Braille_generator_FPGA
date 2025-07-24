`timescale 1ns / 1ps
`include "defines_cnn_core.vh"

// 3채널 fully-connected 누적합(with valid chain)
module cnn_acc_ci(
    input clk,
    input reset_n,

    input i_in_valid,
    input [`CO * `OF_BW-1:0] i_in_pooling,  // 3채널, 34비트씩 flatten

    output o_ot_valid,
    output [`CO * `ACC_BW -1:0] o_ot_ci_acc
);
    // ---------------------
    // 파라미터/신호 정의
    // ---------------------
    reg [$clog2(16)-1:0] cnt;    // 16회 풀링 인덱스(4x4 maxpool)
    reg signed [7:0] rom[0:143]; // 3*48

    // 누적 레지스터(결과 저장)
    reg  [`CO * `ACC_BW-1:0] r_acc;
    reg  [`CO * `ACC_BW-1:0] r_out;
    reg  r_valid;

    wire [`CO * (`MUL_BW + $clog2(3)) - 1:0] w_ot_kernel;
    wire [`CO-1:0] w_ot_valid;

    // -- 가중치 메모리 로딩
    initial $readmemh("fc1_weights.mem", rom);

    // -- pooling index 카운터
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            cnt <= 0;
        else if (&w_ot_valid)
            cnt <= (cnt == 15) ? 0 : cnt + 1;
    end

    // -- cnn_kernal 인스턴스 생성 및 각 채널 w_ot_valid 연결
    genvar mul_inst;
    generate
        for (mul_inst = 0; mul_inst < `CO; mul_inst = mul_inst + 1) begin: gen_mul
            wire [7:0] weight0 = rom[mul_inst*48 + cnt];
            wire [7:0] weight1 = rom[mul_inst*48 + 16 + cnt];
            wire [7:0] weight2 = rom[mul_inst*48 + 32 + cnt];
            wire [`CO*`W_BW-1:0] w_cnn_weight = {weight2, weight1, weight0};
            cnn_kernal U_cnn_kernal(
                .clk(clk),
                .reset_n(reset_n),
                .i_pooling_valid(i_in_valid),
                .i_pooling(i_in_pooling),
                .i_weight(w_cnn_weight),
                .o_kernal_valid(w_ot_valid[mul_inst]),
                .o_kernel(w_ot_kernel[mul_inst * (`MUL_BW + $clog2(3)) +: (`MUL_BW + $clog2(3))])
            );
        end
    endgenerate

    // -- 실제 누적합 연산 : 모든 채널의 valid(&w_ot_valid)이 '1'일 때만 누적!
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            r_acc <= 0;
        else if (&w_ot_valid) begin
            if (cnt == 0)
                r_acc <= w_ot_kernel;  // 첫 번째 pooling: 그냥 대입(누적 시작)
            else
                for (i = 0; i < `CO; i = i + 1)
                    r_acc[i*`ACC_BW+:`ACC_BW] <= $signed(r_acc[i*`ACC_BW+:`ACC_BW])
                                                + $signed(w_ot_kernel[i*(`MUL_BW+$clog2(3))+:(`MUL_BW+$clog2(3))]);
        end
    end

    // -- 최종 valid/latch: 16번째 pooling 입력 + valid 발생 시 결과 출력
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_out <= 0;
            r_valid <= 0;
        end else if (&w_ot_valid && (cnt == 15)) begin
            r_out <= r_acc;   // 최종 누산값
            r_valid <= 1;     // 1클럭 valid
        end else begin
            r_valid <= 0;
        end
    end

    assign o_ot_valid = r_valid;
    assign o_ot_ci_acc = r_out;

endmodule
