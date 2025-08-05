`timescale 1ns / 1ps


`include "defines_cnn_core.v"

// 3채널 fully-connected 누적합(with valid chain)
module stage3_cnn_acc_ci(
    input clk,
    input reset_n,

    input i_in_valid,
    input [`pool_CO * `ST3_OF_BW-1:0] i_in_pooling,  // 3채널, 35비트씩 flatten

    output o_ot_valid,
    output [`acc_CO * `ST3_ACC_BW -1:0] o_ot_ci_acc
);
    // ---------------------
    // 파라미터/신호 정의
    // ---------------------

    reg [$clog2(`P_SIZE * `P_SIZE)-1:0] cnt;    // 16회 풀링 인덱스(4x4 maxpool)
    reg [$clog2(`P_SIZE * `P_SIZE)-1:0] w_cnt;    // 16회 풀링 인덱스(4x4 maxpool)
    reg signed [`ST3_W_BW - 1:0] rom[0:`acc_CO * `pool_CO * `P_SIZE * `P_SIZE - 1]; // 3*48 `acc_CO * `pool_CO * `P_SIZE * `P_SIZE

    // 누적 레지스터(결과 저장)
    reg  [`acc_CO * `ST3_ACC_BW-1:0] acc_kernel;
    reg  [`acc_CO * `ST3_ACC_BW-1:0] r_out;
    reg  r_valid;

    wire [`acc_CO - 1:0] w_ot_valid;
    wire signed [`acc_CO * (`ST3_KER_BW) - 1:0] w_ot_kernel;

    // -- 가중치 메모리 로딩
    initial $readmemh("stage3_fc1_weights.mem", rom);
    reg [`pool_CO*`ST3_W_BW-1:0] w_cnn_weight[0:`acc_CO-1];

    // -- pooling index 카운터

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cnt <= 0;
        end
        else if (&w_ot_valid) begin
            cnt <= (cnt == 15) ? 0 : cnt + 1;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            w_cnt <= 0;
        end
        else if (i_in_valid) begin
            w_cnt <= (w_cnt == 15) ? 0 : w_cnt + 1;
        end
    end

    integer ch, mul;
    always @(*) begin
        for (mul = 0; mul < `acc_CO; mul = mul + 1) begin
            for (ch = 0; ch < `pool_CO; ch = ch + 1 ) begin
                w_cnn_weight [mul][ch* `ST3_W_BW +: `ST3_W_BW] = rom[(mul*`FC_IN_VEC) + (ch * `P_SIZE * `P_SIZE) + w_cnt];
            end 
        end
    end

    // -- cnn_kernal 인스턴스 생성 및 각 채널 w_ot_valid 연결
    genvar mul_inst;
    generate
        for (mul_inst = 0; mul_inst < `acc_CO; mul_inst = mul_inst + 1) begin: gen_mul
            stage3_cnn_kernal U_cnn_kernal(
                .clk(clk),
                .reset_n(reset_n),
                .i_pooling_valid(i_in_valid),
                .i_pooling(i_in_pooling),
                .i_weight(w_cnn_weight[mul_inst]),
                .o_kernal_valid(w_ot_valid[mul_inst]),
                .o_kernel(w_ot_kernel[mul_inst * `ST3_KER_BW +: `ST3_KER_BW])
            );
        end
    endgenerate

    integer i;
    generate
        always @ (posedge clk or negedge reset_n) begin
            if (!reset_n) begin
                acc_kernel <= 0;
            end else if(r_valid) begin
                acc_kernel <= 0;
            end else if (&w_ot_valid) begin
                for (i=0; i< `acc_CO; i = i+1) begin
                    acc_kernel[i * `ST3_ACC_BW +: `ST3_ACC_BW] <= $signed(acc_kernel[i * `ST3_ACC_BW +: `ST3_ACC_BW]) + $signed(w_ot_kernel[i * `ST3_KER_BW +: `ST3_KER_BW]); 
                end
            end
        end
    endgenerate

    // reg  [`CO * `ST3_ACC_BW-1:0] next_acc;
    // reg  [`CO * `ST3_ACC_BW-1:0] r_out;

    // (* mark_debug = "true" *) reg signed [`ST3_ACC_BW-1:0] d_acc [0 : `acc_CO -1];

    // integer ch;
    // always @(posedge clk or negedge reset_n) begin
    //     if (!reset_n) begin
    //             d_acc [ch] <= 0;
    //         end else if(r_valid) begin
    //             d_acc [ch] <= 0;
    //         end else if (&w_ot_valid) begin
    //         for (ch = 0; ch < `acc_CO; ch = ch + 1) begin
    //             d_acc [ch] <= $signed(acc_kernel[ch * `ST3_ACC_BW +: `ST3_ACC_BW]) + $signed(w_ot_kernel[ch * `ST3_KER_BW +: `ST3_KER_BW]);
    //         end
    //     end
    // end
    // ////////////////////////////////////////////////////////////////////////

    // // -- 최종 valid/latch: 16번째 pooling 입력 + valid 발생 시 결과 출력
    // always @(posedge clk or negedge reset_n) begin
    //     if (!reset_n) begin
    //         r_out <= 0;
    //     end else if (&w_ot_valid && (cnt == 15)) begin
    //         r_out <= acc_kernel;   // 최종 누산값
    //     end
    // end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= 0;
        end else if (&w_ot_valid && (cnt == 15)) begin
            r_valid <= 1;     // 1클럭 valid
        end else begin
            r_valid <= 0;
        end
    end    
    

    assign o_ot_valid = r_valid;
    assign o_ot_ci_acc = acc_kernel;

endmodule